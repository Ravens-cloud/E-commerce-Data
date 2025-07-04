# ml_pipelines/supplier_evaluation_pipeline.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, first
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

def feature_engineering(spark):
    """
    从Hive加载数据并为每个供应商构建特征。
    """
    print("开始进行供应商特征工程...")
    orders_df = spark.table("ecom_dw.fact_order")
    vendors_df = spark.table("ecom_dw.dim_vendor")
    products_df = spark.table("ecom_dw.dim_product")
    
    # 计算每个供应商的平均利润率
    vendor_profit_df = products_df.join(vendors_df, "vendor_id") \
        .withColumn("profit_margin", (col("unit_price") - col("cost_price")) / col("cost_price")) \
        .groupBy("vendor_id") \
        .agg(avg("profit_margin").alias("avg_profit_margin"))

    # 聚合供应商的订单表现
    vendor_features_df = orders_df.groupBy("vendor_id").agg(
        count("order_id").alias("sales_volume"),
        sum("net_amount").alias("sales_value"),
        (sum(when(col("order_status") == 'Returned', 1).otherwise(0)) / count("order_id")).alias("order_return_rate")
    )
    
    # 组合所有特征
    features_df = vendors_df.join(vendor_profit_df, "vendor_id", "left") \
                            .join(vendor_features_df, "vendor_id", "left")
    
    features_df = features_df.na.fill(0)

    # **核心：标签生成**
    # 基于业务逻辑创建一个综合表现分作为“真实标签”
    features_df = features_df.withColumn(
        "performance_label",
        (col("vendor_rating") * 0.4) +
        (col("avg_profit_margin") * 0.3) +
        ((1 - col("order_return_rate")) * 0.2) +
        ((1 - col("avg_delivery_time") / 10) * 0.1) 
    )
    
    print("供应商特征工程完成。")
    return features_df

def train_and_evaluate_model(features_df, model_path):
    """
    训练、评估并保存供应商评估模型。
    """
    print("开始训练和评估模型...")
    feature_cols = ['cost_price', 'service_quality', 'avg_delivery_time', 'avg_profit_margin', 'sales_volume', 'sales_value', 'order_return_rate']
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")
    scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
    gbt = GBTRegressor(featuresCol="features", labelCol="performance_label", maxIter=10)
    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    (training_data, test_data) = features_df.randomSplit([0.8, 0.2], seed=42)
    
    model = pipeline.fit(training_data)
    
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="performance_label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"模型在测试集上的RMSE (均方根误差): {rmse:.4f}")

    model.write().overwrite().save(model_path)
    print(f"模型已保存至 HDFS: {model_path}")

def score_and_write_back(spark, features_df, model_path, output_table):
    """
    加载模型对全量供应商进行评分，并写回Hive。
    """
    print("开始对全量供应商进行评分...")
    persisted_model = PipelineModel.load(model_path)
    full_predictions = persisted_model.transform(features_df)
    
    vendor_scores = full_predictions.select(
        col("vendor_id"),
        col("vendor_name"),
        col("prediction").alias("performance_score")
    ).withColumn("scoring_date", lit("2025-07-01"))
    
    print(f"准备将评分结果写回Hive表: {output_table}")
    vendor_scores.write.mode("overwrite").format("orc").saveAsTable(output_table)
    print("评分结果回写成功。")

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("SupplierEvaluationPipeline") \
        .enableHiveSupport() \
        .getOrCreate()
        
    MODEL_PATH = "/models/supplier_eval_model/v1"
    OUTPUT_TABLE = "ecom_dw.vendor_performance_score"

    all_features = feature_engineering(spark)
    all_features.cache()
    
    train_and_evaluate_model(all_features, MODEL_PATH)
    score_and_write_back(spark, all_features, MODEL_PATH, OUTPUT_TABLE)
    
    all_features.unpersist()
    spark.stop()