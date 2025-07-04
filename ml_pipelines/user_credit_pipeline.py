# ml_pipelines/user_credit_pipeline.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, datediff, lit, max, when, udf
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def feature_engineering(spark, end_date_str):
    """
    从Hive加载数据并为每个用户构建特征集。
    """
    print("开始进行特征工程...")

    orders_df = spark.table("ecom_dw.fact_order")
    users_df = spark.table("ecom_dw.dim_user")


    features_df = orders_df.groupBy("user_id").agg(
        max("order_date").alias("last_purchase_date"),
        count("order_id").alias("frequency"),
        sum("net_amount").alias("monetary"),
        avg("discount_amount").alias("avg_discount"),
        (sum(when(col("order_status") == 'Returned', 1).otherwise(0)) / count("order_id")).alias("return_rate_model"),
        count(when(col("payment_method") == 'prepaid_card', 1)).alias("prepaid_card_count")
    ).join(users_df.select("user_id", "registration_date", "credit_score"), "user_id", "right") \
     .withColumn("recency", datediff(lit(end_date_str), col("last_purchase_date"))) \
     .withColumn("user_tenure", datediff(lit(end_date_str), col("registration_date")))

    
    features_df = features_df.na.fill(0, subset=['frequency', 'monetary', 'avg_discount', 'return_rate_model', 'prepaid_card_count'])
    features_df = features_df.na.fill(999, subset=['recency']) # 新用户没有购买记录，recency设为大数

    # **核心：标签生成**
    # 这是一个业务定义的简化规则。在真实场景中，标签应来自已知的欺诈事件。
    features_df = features_df.withColumn(
        "label",
        when((col("return_rate_model") > 0.6) & (col("prepaid_card_count") > 1), 1.0)
        .when(col("credit_score") < 400, 1.0)
        .when((col("recency") < 10) & (col("monetary") > 5000), 1.0) # 短期内高额消费
        .otherwise(0.0)
    )
    
    print("特征工程完成。")
    return features_df

def train_and_evaluate_model(features_df, model_path):
    """
    训练、评估并保存模型。
    """
    print("开始训练和评估模型...")
    feature_cols = ['frequency', 'monetary', 'recency', 'avg_discount', 'return_rate_model', 'user_tenure', 'prepaid_card_count']
    
    # ML Pipeline定义
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
    rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label", numTrees=100, seed=42)
    pipeline = Pipeline(stages=[assembler, scaler, rf])

    
    (training_data, test_data) = features_df.randomSplit([0.8, 0.2], seed=42)
    
    model = pipeline.fit(training_data)

    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"模型在测试集上的AUC: {auc:.4f}")

    model.write().overwrite().save(model_path)
    print(f"模型已保存至 HDFS: {model_path}")
    
def score_and_write_back(spark, features_df, model_path, output_table):
    """
    加载模型对全量用户进行评分，并将结果写回Hive。
    """
    print("开始对全量用户进行评分...")
    persisted_model = PipelineModel.load(model_path)
    
    full_predictions = persisted_model.transform(features_df)
    
    # UDF: 从概率向量中提取高风险(标签=1)的概率
    extract_prob_udf = udf(lambda v: float(v[1]), FloatType())

    user_scores = full_predictions.select(
        col("user_id"),
        col("prediction").alias("is_high_risk"), # 0或1的最终预测
        extract_prob_udf("probability").alias("risk_score") # 0到1的风险概率
    ).withColumn("scoring_date", lit(end_date_str))
    
    print(f"准备将评分结果写回Hive表: {output_table}")
    user_scores.write.mode("overwrite").format("orc").saveAsTable(output_table)
    print("评分结果回写成功。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_date", default="2025-07-01", help="数据快照的结束日期")
    args = parser.parse_args()
    end_date_str = args.end_date

    spark = SparkSession.builder \
        .appName("UserCreditScoringPipeline") \
        .enableHiveSupport() \
        .getOrCreate()
    
    MODEL_PATH = "/models/user_credit_model/v1"
    OUTPUT_TABLE = "ecom_dw.user_credit_score"
    
    all_features = feature_engineering(spark, end_date_str)
    all_features.cache() # 缓存以加速后续步骤
    
    train_and_evaluate_model(all_features, MODEL_PATH)
    score_and_write_back(spark, all_features, MODEL_PATH, OUTPUT_TABLE)
    
    all_features.unpersist()
    spark.stop()