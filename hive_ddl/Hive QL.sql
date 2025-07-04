-- =================================================================
-- Hive QL for Business Analysis & Decision Support
-- =================================================================

-- 1. 核心指标分析: 商品品类销售额及占比 (用于可视化看板)
-- 目标：分析各商品大类的销售表现，支撑品类运营决策。
SELECT
    SPLIT(p.category, '>')[0] AS main_category,
    SUM(o.net_amount) AS total_sales,
    ROUND(SUM(o.net_amount) * 100 / SUM(SUM(o.net_amount)) OVER (), 2) AS sales_percentage
FROM fact_order o
JOIN dim_product p ON o.product_id = p.product_id
WHERE o.order_date BETWEEN '2025-04-01' AND '2025-06-30' -- 按季度分析
GROUP BY SPLIT(p.category, '>')[0]
ORDER BY total_sales DESC;


-- 2. 用户复购周期分析
-- 目标：计算用户平均复购间隔，为提升复购率提供数据支持。
WITH user_purchase_lag AS (
    SELECT
        user_id,
        order_date,
        LAG(order_date, 1, NULL) OVER (PARTITION BY user_id ORDER BY order_date) AS last_purchase_date
    FROM (
        -- 对同一用户同一天的订单进行去重
        SELECT DISTINCT user_id, order_date FROM fact_order
    ) daily_purchases
)
SELECT
    -- 计算所有复购用户（至少购买两次）的平均复购间隔天数
    ROUND(AVG(DATEDIFF(order_date, last_purchase_date)), 2) AS avg_repurchase_cycle_days
FROM user_purchase_lag
WHERE last_purchase_date IS NOT NULL;


-- 3. 优惠券策略有效性分析 (ROI分析)
-- 目标：评估优惠券对销售额的提升效果，优化营销成本。
SELECT
    d.discount_name,
    d.discount_type,
    COUNT(o.order_id) AS orders_with_coupon,
    SUM(o.net_amount) AS sales_with_coupon,
    SUM(o.discount_amount) AS total_discount_cost,
    -- 计算ROI：(带来的销售额 - 成本) / 成本
    ROUND((SUM(o.net_amount) - SUM(o.discount_amount)) / SUM(o.discount_amount), 2) AS coupon_roi
FROM fact_order o
JOIN dim_discount d ON o.discount_id = d.discount_id
WHERE o.discount_id IS NOT NULL
GROUP BY d.discount_name, d.discount_type
ORDER BY coupon_roi DESC;


-- 4. 供应商性价比评估模型 (筛选Top 10优质供应商)
-- 目标：识别性价比最高的供应商，实现降本增效。
SELECT
    v.vendor_id,
    v.vendor_name,
    v.vendor_rating,
    v.vendor_return_rate,
    -- 计算指标：售出商品总成本与总售价
    SUM(o.quantity * v.cost_price) AS total_cost,
    SUM(o.gross_amount) AS total_gross_sales,
    -- 核心性价比评分 (利润率越高、退货率越低、评级越高，得分越高)
    ROUND(
        (SUM(o.gross_amount) - SUM(o.quantity * v.cost_price)) / SUM(o.quantity * v.cost_price) * 0.5 +
        v.vendor_rating * 0.3 +
        (1 - v.vendor_return_rate) * 0.2, 4
    ) AS value_score
FROM fact_order o
JOIN dim_vendor v ON o.vendor_id = v.vendor_id
GROUP BY v.vendor_id, v.vendor_name, v.vendor_rating, v.vendor_return_rate
ORDER BY value_score DESC
LIMIT 10;


-- 5. 数据驱动的仓库选址方案分析
-- 目标：识别订单需求最集中的地理区域，为仓库选址提供依据，降低缺货率。
SELECT
    province,
    city,
    COUNT(order_id) AS total_orders,
    SUM(quantity) AS total_quantity,
    SUM(net_amount) AS total_sales
FROM fact_order
WHERE order_date >= '2025-01-01' -- 分析一段时期内的数据
GROUP BY province, city
ORDER BY total_orders DESC, total_sales DESC
LIMIT 20; -- 找出需求量最大的20个城市


-- 6. 用户信用评级体系 (初步构建)
-- 目标：识别高风险用户，减少欺诈订单。
WITH user_metrics AS (
    SELECT
        u.user_id,
        -- 近90天内是否有退货行为
        MAX(CASE WHEN o.order_status = 'Returned' THEN 1 ELSE 0 END) AS has_returned_recently,
        -- 总消费金额
        SUM(o.net_amount) AS total_spend,
        -- 平均订单金额
        AVG(o.net_amount) AS avg_order_value,
        -- 成为会员的天数
        DATEDIFF(CURRENT_DATE(), u.registration_date) AS tenure_days
    FROM dim_user u
    JOIN fact_order o ON u.user_id = o.user_id
    WHERE o.order_date > DATE_SUB(CURRENT_DATE(), 90)
    GROUP BY u.user_id, u.registration_date
)
SELECT
    user_id,
    -- 构建一个简单的信用分数 (总消费越高、客单价越高、会员时间越长、无退货 -> 分数越高)
    CASE
        WHEN has_returned_recently = 1 THEN 'Low'
        WHEN total_spend > 5000 AND avg_order_value > 500 AND tenure_days > 180 THEN 'High'
        WHEN total_spend > 1000 AND tenure_days > 90 THEN 'Medium'
        ELSE 'Standard'
    END AS credit_rating
FROM user_metrics
ORDER BY total_spend DESC;