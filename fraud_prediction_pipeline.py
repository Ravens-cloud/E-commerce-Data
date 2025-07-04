# fraud_prediction_pipeline.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, datediff, lit, max, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import PipelineModel


spark = SparkSession.builder \
    .appName("UserCreditScoringPipeline") \
    .enableHiveSupport() \
    .getOrCreate()


orders_df = spark.table("ecom_dw.fact_order")
users_df = spark.table("ecom_dw.dim_user")

# 2. 特征工程 (为每个user_id聚合特征)
feature_df = orders_df.groupBy("user_id").agg(
    # RFM
    max("order_date").alias("last_purchase_date"),
    count("order_id").alias("frequency"),
    sum("net_amount").alias("monetary"),
    # 行为特征
    avg("discount_amount").alias("avg_discount"),
    (sum(when(col("order_status") == 'Returned', 1).otherwise(0)) / count("order_id")).alias("return_rate_model"),
    count(when(col("payment_method") == 'prepaid_card', 1)).alias("prepaid_card_count")
).join(users_df.select("user_id", "registration_date", "vip_level", "credit_score"), "user_id") \
 .withColumn("recency", datediff(lit("2025-07-01"), col("last_purchase_date"))) \
 .withColumn("user_tenure", datediff(lit("2025-07-01"), col("registration_date")))

# 3. 标签生成 (核心业务逻辑)
# 定义：高退货率(>0.5)且使用预付卡的，或历史信用分低的，定义为高风险(1)
# 注意：这只是一个示例，实际标签应来自业务历史数据（如真实欺诈报告）
feature_df = feature_df.withColumn(
    "label",
    when((col("return_rate_model") > 0.5) & (col("prepaid_card_count") > 0), 1.0)
    .when(col("credit_score") < 450, 1.0)
    .otherwise(0.0)
)
feature_df = feature_df.na.fill(0) # 填充空值

print("特征工程与标签生成完成。")
feature_df.show(5)

feature_cols = ['frequency', 'monetary', 'recency', 'avg_discount', 'return_rate_model', 'user_tenure', 'vip_level']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label", numTrees=100)

pipeline = Pipeline(stages=[assembler, scaler, rf])

(training_data, test_data) = feature_df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(training_data)
print("模型训练完成。")

predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"模型在测试集上的AUC: {auc:.4f}")


model_path = "/models/user_credit_model/v1"
model.write().overwrite().save(model_path)
print(f"模型已保存至 HDFS: {model_path}")


all_users_features = ... # (此处的特征工程与步骤2类似，但不包含label列)
all_users_features = assembler.transform(all_users_features)
all_users_features = scaler_model.transform(all_users_features) # 注意：用fit好的scaler model


persisted_model = PipelineModel.load(model_path)


full_predictions = persisted_model.transform(all_users_features)

# 提取 user_id 和 预测概率/分数
# probability是一个向量[P(0), P(1)]，我们取P(1)作为风险分数
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

extract_prob_udf = udf(lambda v: float(v[1]), FloatType())

user_scores = full_predictions.select(
    "user_id",
    "prediction", # 0或1的最终预测
    extract_prob_udf("probability").alias("risk_score") # 0到1的风险概率
)

print("全量用户评分完成。")
user_scores.show(5)


user_scores.write.mode("overwrite").format("orc").saveAsTable("ecom_dw.user_credit_score")

print("用户信用评分已成功回写到Hive表 'ecom_dw.user_credit_score'。")
spark.stop()