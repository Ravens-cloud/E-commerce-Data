-- =================================================================
-- Hive DWH DDL Script for E-commerce Intelligent Decision System
-- Database: ecom_dw
-- =================================================================

CREATE DATABASE IF NOT EXISTS ecom_dw;
USE ecom_dw;

-- ================================================
-- 维度表 (Dimension Tables)
-- ================================================

-- 1. 用户维表 (dim_user)
CREATE EXTERNAL TABLE IF NOT EXISTS dim_user (
  user_id               BIGINT COMMENT '用户唯一标识',
  user_name             STRING COMMENT '用户名',
  gender                CHAR(1) COMMENT '性别: M, F, O',
  age                   SMALLINT COMMENT '年龄',
  email                 STRING COMMENT '电子邮箱',
  phone                 STRING COMMENT '电话号码',
  province              STRING COMMENT '省份/州',
  city                  STRING COMMENT '城市',
  registration_date     DATE COMMENT '注册日期',
  last_login            TIMESTAMP COMMENT '最后登录时间',
  vip_level             TINYINT COMMENT 'VIP等级',
  discount_sensitivity  DECIMAL(5,4) COMMENT '折扣敏感度 (0-1)',
  return_rate           DECIMAL(5,4) COMMENT '历史退货率 (0-1)',
  credit_score          INT COMMENT '用户信用评分 (300-850)'
)
PARTITIONED BY (country STRING COMMENT '国家')
CLUSTERED BY (user_id) INTO 32 BUCKETS
STORED AS ORC
TBLPROPERTIES (
  "orc.compress"="ZLIB",
  "comment"="用户维度信息表"
);

-- 2. 商品维表 (dim_product)
CREATE EXTERNAL TABLE IF NOT EXISTS dim_product (
  product_id            BIGINT COMMENT '商品唯一标识',
  category              STRING COMMENT '商品品类 (e.g., 服装>男装)',
  brand                 STRING COMMENT '品牌名称',
  unit_price            DECIMAL(10,2) COMMENT '单价',
  stock                 INT COMMENT '库存数量',
  description           STRING COMMENT '商品描述',
  implicit_attributes   STRING COMMENT '隐式属性 (JSON格式: e.g., {"color":"red"})',
  vendor_id             BIGINT COMMENT '供应商ID'
)
CLUSTERED BY (product_id) INTO 16 BUCKETS
STORED AS ORC
TBLPROPERTIES (
  "orc.compress"="ZLIB",
  "comment"="商品维度信息表"
);

-- 3. 供应商维表 (dim_vendor)
CREATE EXTERNAL TABLE IF NOT EXISTS dim_vendor (
  vendor_id             BIGINT COMMENT '供应商唯一标识',
  vendor_name           STRING COMMENT '供应商名称',
  cost_price            DECIMAL(10,2) COMMENT '成本价',
  market_price          DECIMAL(10,2) COMMENT '市场价',
  vendor_rating         DECIMAL(3,2) COMMENT '供应商综合评级 (1-5)',
  service_quality       DECIMAL(3,2) COMMENT '服务质量评分 (1-5)',
  avg_delivery_time     INT COMMENT '平均发货天数',
  vendor_return_rate    DECIMAL(5,4) COMMENT '供应商退货率',
  distribution_areas    ARRAY<STRING> COMMENT '主要配送区域'
)
PARTITIONED BY (country STRING COMMENT '供应商所在国家')
STORED AS ORC
TBLPROPERTIES (
  "orc.compress"="ZLIB",
  "comment"="供应商维度信息表"
);

-- 4. 折扣维表 (dim_discount)
CREATE EXTERNAL TABLE IF NOT EXISTS dim_discount (
  discount_id           BIGINT COMMENT '折扣唯一标识',
  discount_name         STRING COMMENT '折扣活动名称',
  discount_type         STRING COMMENT '折扣类型 (fixed, coupon, percentage)',
  threshold_amount      DECIMAL(12,2) COMMENT '使用门槛金额',
  discount_value        DECIMAL(10,2) COMMENT '折扣值 (固定金额或百分比)',
  valid_from            TIMESTAMP COMMENT '生效时间',
  valid_to              TIMESTAMP COMMENT '失效时间'
)
STORED AS ORC
TBLPROPERTIES (
  "orc.compress"="ZLIB",
  "comment"="折扣维度信息表"
);

-- 5. 时间维表 (dim_time)
CREATE EXTERNAL TABLE IF NOT EXISTS dim_time (
  time_id               INT COMMENT '时间唯一ID (YYYYMMDD)',
  dt                    DATE COMMENT '日期',
  year                  SMALLINT,
  quarter               TINYINT,
  month                 TINYINT,
  day                   TINYINT,
  hour                  TINYINT,
  minute                TINYINT,
  day_of_week           TINYINT COMMENT '周几 (1=周一)',
  is_weekend            TINYINT COMMENT '是否周末 (1=是, 0=否)'
)
STORED AS PARQUET -- 时间维表查询频繁，Parquet列式存储性能更佳
TBLPROPERTIES (
  "comment"="时间维度信息表"
);

-- ================================================
-- 事实表 (Fact Tables)
-- ================================================

-- 6. 订单事实表 (fact_order) - 核心事务型事实表
CREATE EXTERNAL TABLE IF NOT EXISTS fact_order (
  order_id              STRING COMMENT '订单唯一标识',
  user_id               BIGINT COMMENT '用户ID',
  product_id            BIGINT COMMENT '商品ID',
  discount_id           BIGINT COMMENT '折扣ID',
  time_id               INT COMMENT '时间ID (关联dim_time)',
  vendor_id             BIGINT COMMENT '供应商ID',
  quantity              INT COMMENT '购买数量',
  gross_amount          DECIMAL(12,2) COMMENT '订单总额 (折扣前)',
  discount_amount       DECIMAL(12,2) COMMENT '折扣金额',
  net_amount            DECIMAL(12,2) COMMENT '实付金额',
  payment_method        STRING COMMENT '支付方式',
  purchase_channel      STRING COMMENT '购买渠道',
  order_status          STRING COMMENT '订单状态',
  province              STRING COMMENT '收货省份',
  city                  STRING COMMENT '收货城市'
)
PARTITIONED BY (order_date DATE COMMENT '订单日期分区')
CLUSTERED BY (user_id) INTO 64 BUCKETS
STORED AS ORC
TBLPROPERTIES (
  "orc.compress"="ZLIB",
  "comment"="订单事实表"
);

-- 7. 用户行为事件事实表 (fact_event) - 累积快照事实表
CREATE EXTERNAL TABLE IF NOT EXISTS fact_event (
  event_id              STRING COMMENT '事件唯一ID',
  user_id               BIGINT COMMENT '用户ID',
  product_id            BIGINT COMMENT '商品ID (若相关)',
  session_id            STRING COMMENT '会话ID',
  event_type            STRING COMMENT '事件类型 (view, cart_add, purchase)',
  event_ts              TIMESTAMP COMMENT '事件发生精确时间'
)
PARTITIONED BY (event_date DATE COMMENT '事件日期分区')
STORED AS ORC
TBLPROPERTIES (
  "orc.compress"="ZLIB",
  "comment"="用户行为事件日志表"
);


-- 8. 订单事实表
CREATE TABLE IF NOT EXISTS fact_order (
  order_id         BIGINT       NOT NULL,
  user_id          BIGINT       NOT NULL,
  product_id       BIGINT       NOT NULL,
  time_id          BIGINT       NOT NULL,
  vendor_id        BIGINT       NOT NULL,
  quantity         INT          NOT NULL  CHECK (quantity > 0),
  gross_amount     DECIMAL(12,2)NOT NULL  CHECK (gross_amount > 0),
  discount_amount  DECIMAL(12,2)NOT NULL  CHECK (discount_amount >= 0),
  discount_rate    DECIMAL(5,4)  GENERATED ALWAYS AS (discount_amount/gross_amount),
  net_amount       DECIMAL(12,2)NOT NULL  CHECK (net_amount >= 0),
payment_method   STRING       NOT NULL  CHECK (
    payment_method IN (
      'credit_card',      
      'third_party',      
      'prepaid_card',     
      'wire_transfer',    
      'unionpay'          
    )
  ),
  purchase_channel STRING       NOT NULL  CHECK (
    purchase_channel IN (
      'web',           
      'mobile_app',     
      'mini_program',   
    )
  province         STRING,
  city             STRING
)
PARTITIONED BY (year SMALLINT, month TINYINT)
CLUSTERED BY (user_id) INTO 10 BUCKETS
STORED AS ORC
TBLPROPERTIES ("orc.compress"="ZLIB");


-- ================================================
-- 辅助画像/策略表（Optional）
-- ================================================

-- 9. 用户行为分析辅助表
CREATE TABLE IF NOT EXISTS user_behavior_profile (
  user_id               BIGINT       NOT NULL,
  avg_research_time     INT          CHECK (avg_research_time > 0),
  decision_time         INT          CHECK (decision_time > 0),
  purchase_intensity    DECIMAL(5,2) CHECK (purchase_intensity BETWEEN 0 AND 1),
  preferred_category    STRING,
  channel_preference    ARRAY<STRING>,
  social_influence_factor DECIMAL(5,2) CHECK (social_influence_factor BETWEEN 0 AND 1)
)
STORED AS PARQUET;

-- 10. 优惠券推送策略表
CREATE TABLE IF NOT EXISTS coupon_strategy (
  strategy_id           BIGINT       NOT NULL,
  min_discount_sensitivity DECIMAL(5,2) NOT NULL  CHECK (min_discount_sensitivity BETWEEN 0 AND 1),
  min_view_count        INT          NOT NULL  CHECK (min_view_count >= 0),
  min_cart_count        INT          NOT NULL  CHECK (min_cart_count >= 0),
  trigger_type          STRING       NOT NULL  CHECK (trigger_type IN ('view', 'cart_add')), 
  action_type           STRING       NOT NULL  CHECK (action_type IN ('popup', 'push', 'sms')),
  coupon_id             BIGINT       NOT NULL,
)
STORED AS ORC
TBLPROPERTIES ("orc.compress"="ZLIB");