# 基于Hadoop的电商智能决策分析系统

本项目旨在构建一个端到端的、基于Hadoop生态的大数据分析与决策系统。系统通过模拟海量电商用户行为数据，构建企业级数据仓库，并利用数据分析和机器学习模型，为业务决策提供数据支持，实现从数据到价值的转化。

## 项目架构

系统遵循经典的大数据处理架构，分为数据生成、数据存储与建模、数据分析与应用三个核心层次，并通过自动化的MLOps流程形成闭环。

```
+--------------------------+     +--------------------------+     +--------------------------+
|    数据生成与采集        | --> |  数据存储、建模与处理    | --> |    数据分析与智能应用    |
| (Python, Faker, Scipy)   |     |  (HDFS, Hive, Spark)     |     |  (Hive QL, Spark ML)     |
+--------------------------+     +--------------------------+     +--------------------------+
           |                                ^                                |
           |________________________________|________________________________|
                                    |
                          +------------------------+
                          |   自动化调度与监控     |
                          |   (Airflow, Oozie)     |
                          +------------------------+
```

## 功能模块

1.  **数据模拟**: 使用Python脚本(`generate_ecommerce_data.py`)生成符合业务逻辑的、标准化的多维数据，包括用户、商品、订单、行为日志等。
2.  **数据仓库**: 在Hive中构建了采用星型模型的数据仓库，通过分区和分桶技术优化查询性能，支持TB级数据存储与分析。
3.  **商业智能(BI)分析**: 提供了一系列核心Hive QL查询，用于常规的业务分析，如销售额统计、品类排行、用户复购周期分析等。
4.  **高级分析与机器学习**:
    * **用户信用评级模型**: 训练一个分类模型，预测高风险或欺诈用户，减少业务损失。
    * **供应商评估模型**: 训练一个回归模型，对供应商进行综合评分，优化供应链管理，实现降本增效。

## 目录结构

```
.
├── ecommerce_data/                 # [生成] 模拟生成的CSV数据
├── fraud_prediction_pipeline.py  
├── hive_ddl/                       # Hive DDL建表语句
│   └── create_tables.sql
│   └── Hive QL.sql
├── ml_pipelines/                   # PySpark机器学习流程脚本
│   ├── user_credit_scoring.py
│   └── supplier_performance_model.py
├── python_datagen/                 # Python数据生成器
│   └── generate_ecommerce_data.py
└── README.md                       # 项目说明文档
```

## 环境与依赖

* **Hadoop生态**: HDFS, YARN, Hive 3.x, Spark 3.x
* **Python**: 3.7+
* **Python库**: pandas, numpy, faker, scipy, pyspark。通过`pip`安装：
    ```bash
    pip install pandas numpy faker scipy pyspark
    ```

## 运行步骤

#### 第1步: 生成模拟数据

在项目根目录执行Python脚本，生成所有CSV文件到`ecommerce_data/`目录。

```bash
python python_datagen/generate_ecommerce_data.py
```

#### 第2步: 准备HDFS并创建Hive表

1.  **上传数据到HDFS**:
    ```bash
    # 在HDFS上创建数据目录
    hdfs dfs -mkdir -p /user/hive/warehouse/ecom_dw_raw

    # 上传CSV文件
    hdfs dfs -put ecommerce_data/*.csv /user/hive/warehouse/ecom_dw_raw/
    ```

2.  **执行Hive DDL创建数仓表**:
    ```bash
    hive -f hive_ddl/create_tables.sql
    ```
    *注意: 此脚本会创建数据库`ecom_dw`及所有维表和事实表。*

3.  **加载数据到Hive表**:
    由于数据和表已创建，您需要手动或通过脚本将HDFS上的数据加载到对应的分区表中。例如：
    ```sql
    -- 示例：加载用户数据
    LOAD DATA INPATH '/user/hive/warehouse/ecom_dw_raw/dim_user.csv' OVERWRITE INTO TABLE ecom_dw.dim_user PARTITION(country='China');
    -- (需要根据数据中的国家分区重复执行)

    -- 示例：加载订单数据
    ALTER TABLE ecom_dw.fact_order ADD IF NOT EXISTS PARTITION(order_date='2025-01-01');
    LOAD DATA INPATH '/user/hive/warehouse/ecom_dw_raw/fact_order.csv' OVERWRITE INTO TABLE ecom_dw.fact_order PARTITION(order_date='2025-01-01');
    -- (需要为每个日期分区重复执行)
    ```
    *在生产环境中，建议使用Spark或Hive的动态分区功能来自动化此过程。*

#### 第3步: 运行机器学习模型

使用`spark-submit`命令执行ML脚本。确保您的Spark环境可以访问Hive Metastore。

1.  **执行用户信用评分模型**:
    ```bash
    spark-submit \
      --master yarn \
      --deploy-mode client \
      --num-executors 4 \
      --executor-memory 2G \
      ml_pipelines/user_credit_scoring.py
    ```
    执行成功后，模型将被保存在HDFS的`/models/user_credit_model/latest`，评分结果会写入Hive表`ecom_dw.user_credit_score`。

2.  **执行供应商评估模型**:
    ```bash
    spark-submit \
      --master yarn \
      --deploy-mode client \
      --num-executors 4 \
      --executor-memory 2G \
      ml_pipelines/supplier_performance_model.py
    ```
    执行成功后，模型将被保存在HDFS的`/models/supplier_performance_model/latest`，评分结果会写入Hive表`ecom_dw.vendor_performance_score`。

#### 第4步: 在Hive中查询与使用模型结果

模型评分已回写到数仓，您可以直接通过SQL进行查询和分析。

```sql
-- 查询风险分数最高的前20名用户
SELECT * FROM ecom_dw.user_credit_score ORDER BY risk_score DESC LIMIT 20;

-- 查询表现评分最高的前10名供应商
SELECT
    v.vendor_name,
    s.predicted_performance_score
FROM ecom_dw.vendor_performance_score s
JOIN ecom_dw.dim_vendor v ON s.vendor_id = v.vendor_id
ORDER BY s.predicted_performance_score DESC
LIMIT 10;
```

## 未来方向

* **实时化**: 引入Kafka和Spark Streaming，实现用户行为的实时采集和欺诈的实时预警。
* **BI集成**: 将Hive数仓对接到Tableau, Superset等BI工具，创建交互式的数据看板。
* **模型优化**: 使用更复杂的算法（如XGBoost），并通过超参数调优（如Hyperopt）提升模型性能。
* **工作流编排**: 使用Apache Airflow将整个数据处理和模型训练流程编排成自动化的DAG工作流。