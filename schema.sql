CREATE SCHEMA IF NOT EXISTS db_419;
USE db_419;

SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;

-- ================================================
-- Dimension Tables
-- ================================================

-- 1. 用户维表（TEXTFILE格式，动态分区）
CREATE TABLE IF NOT EXISTS dim_user (
    user_id               BIGINT,
    user_name             STRING,
    gender                CHAR(1),
    age                   SMALLINT,
    email                 STRING,
    phone                 STRING,
    province              STRING,
    city                  STRING,
    registration_ts       TIMESTAMP,
    last_login_ts         TIMESTAMP,
    vip_level             TINYINT,
    discount_sensitivity  DECIMAL(5,2),
    return_rate           DECIMAL(5,2),
    credit_score          DECIMAL(5,2),
    user_channel_pref     ARRAY<STRING>
)
PARTITIONED BY (country STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
COLLECTION ITEMS TERMINATED BY '|'
STORED AS TEXTFILE;

-- 2. 商品维表（TEXTFILE格式，OpenCSVSerde）
CREATE TABLE IF NOT EXISTS dim_product (
    product_id           BIGINT,
    category             STRING,
    brand                STRING,
    vendor_id            BIGINT,
    unit_price           DECIMAL(10,2),
    stock_qty            INT,
    description          STRING,
    implicit_attributes  MAP<STRING,STRING>
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    'separatorChar' = ',',
    'quoteChar' = '"',
    'mapkey.delimiter' = ':',
    'colelction.delimiter' = ';'
)
STORED AS TEXTFILE;

-- 3. 供应商维表（TEXTFILE格式，动态分区）
CREATE TABLE IF NOT EXISTS dim_vendor (
    vendor_id           BIGINT,
    vendor_name         STRING,
    region              STRING,
    cost_price          DECIMAL(10,2),
    market_price        DECIMAL(10,2),
    vendor_rating       DECIMAL(3,2),
    service_quality     DECIMAL(3,2),
    avg_delivery_time   INT,
    vendor_return_rate  DECIMAL(5,2),
    distribution_areas  ARRAY<STRING>
)
PARTITIONED BY (country STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
COLLECTION ITEMS TERMINATED BY '|'
STORED AS TEXTFILE;

-- 4. 折扣维表（TEXTFILE格式，无分区）
CREATE TABLE IF NOT EXISTS dim_discount (
    discount_id        BIGINT,
    discount_name      STRING,
    discount_type      STRING,
    threshold_amount   DECIMAL(12,2),
    discount_value     DECIMAL(5,2),
    valid_from_ts      TIMESTAMP,
    valid_to_ts        TIMESTAMP
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;

-- 5. 时间维表（TEXTFILE格式）
CREATE TABLE IF NOT EXISTS dim_time (
    time_id   BIGINT,
    dt        TIMESTAMP,
    year      SMALLINT,
    quarter   TINYINT,
    month     TINYINT,
    day       TINYINT,
    hour      TINYINT,
    minute    TINYINT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;

-- 6. 会话维表（TEXTFILE格式，动态分区）
CREATE TABLE IF NOT EXISTS dim_session (
    session_id   STRING,
    user_id      BIGINT,
    start_ts     TIMESTAMP,
    end_ts       TIMESTAMP,
    device_type  STRING,
    channel      STRING,
    referrer     STRING
)
PARTITIONED BY (session_date DATE)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;

-- ================================================
-- Fact Tables
-- ================================================

-- 7. 用户行为事件事实表（TEXTFILE格式，动态分区）
CREATE TABLE IF NOT EXISTS fact_event (
    event_id         BIGINT,
    user_id          BIGINT,
    product_id       BIGINT,
    time_id          BIGINT,
    session_id       STRING,
    device_type      STRING,
    ad_campaign_id   STRING,
    event_type       STRING,
    event_ts         TIMESTAMP,
    page_duration    INT,
    referrer_source  STRING,
    discount_used    BOOLEAN,
    discount_id      BIGINT,
    event_attributes MAP<STRING,STRING>
)
PARTITIONED BY (event_date DATE)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
COLLECTION ITEMS TERMINATED BY ';'
MAP KEYS TERMINATED BY ':'
STORED AS TEXTFILE;

-- 8. 订单事实表（TEXTFILE格式）
CREATE TABLE IF NOT EXISTS fact_order (
    order_id         BIGINT,
    user_id          BIGINT,
    product_id       BIGINT,
    time_id          BIGINT,
    vendor_id        BIGINT,
    quantity         INT,
    gross_amount     DECIMAL(12,2),
    discount_amount  DECIMAL(12,2),
    net_amount       DECIMAL(12,2),
    payment_method   STRING,
    coupon_type      STRING,
    ad_campaign_id   STRING,
    event_ts         TIMESTAMP,
    purchase_channel STRING,
    order_status     TINYINT,
    province         STRING,
    city             STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;

-- 9. 用户行为分析辅助表（TEXTFILE格式，动态分区）
CREATE TABLE IF NOT EXISTS user_behavior_profile (
    user_id                   BIGINT,
    avg_research_time         INT,
    decision_time             INT,
    purchase_intensity        DECIMAL(5,2),
    preferred_category        STRING,
    channel_preference        ARRAY<STRING>,
    social_influence_factor   DECIMAL(5,2)
)
PARTITIONED BY (profile_date DATE)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
COLLECTION ITEMS TERMINATED BY '|'
STORED AS TEXTFILE;

-- 10. 优惠券推送策略表（TEXTFILE格式）
CREATE TABLE IF NOT EXISTS coupon_strategy (
    strategy_id             BIGINT,
    user_id                 BIGINT,
    min_discount_sensitivity DECIMAL(5,2),
    min_view_count          INT,
    min_cart_count          INT,
    trigger_type            STRING,
    action_type             STRING,
    coupon_id               BIGINT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;