from email.policy import default
import torch
import pandas as pd
import numpy as np
from faker import Faker
import uuid
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime, timedelta
from scipy.stats import beta, poisson, norm, zipf
import re

# ================================================
# Load global configuration from JSON
# ================================================
# with open('config.json', 'r') as _f:
#     CONFIG = json.load(_f)
CONFIG = {
    # 各国主力市场省/州/地区分布及权重（总和为1.0）
    "geo": {
        "China": {
            "provinces": [
                {"name": "广东", "weight": 0.18},
                {"name": "江苏", "weight": 0.12},
                {"name": "浙江", "weight": 0.10},
                {"name": "山东", "weight": 0.08},
                {"name": "河南", "weight": 0.06},
                {"name": "其它省份", "weight": 0.46}
            ],
            "province_city_map": {
                "广东": ["广州", "深圳", "佛山", "东莞"],
                "江苏": ["南京", "苏州", "无锡", "常州"],
                "浙江": ["杭州", "宁波", "温州", "绍兴"]
            }
        },
        "USA": {
            "provinces": [
                {"name": "California", "weight": 0.15},
                {"name": "Texas", "weight": 0.12},
                {"name": "New York", "weight": 0.10},
                {"name": "Florida", "weight": 0.08},
                {"name": "Illinois", "weight": 0.05},
                {"name": "Other States", "weight": 0.50}
            ],
            "province_city_map": {
                "California": ["Los Angeles", "San Francisco", "San Diego"],
                "Texas": ["Houston", "Dallas", "Austin"],
                "New York": ["New York City", "Buffalo", "Rochester"]
            }
        },
        "UK": [
            {"name": "England", "weight": 0.84},
            {"name": "Scotland", "weight": 0.08},
            {"name": "Wales", "weight": 0.05},
            {"name": "Northern Ireland", "weight": 0.03}
        ],
        "Germany": [
            {"name": "北莱茵-威斯特法伦", "weight": 0.21},
            {"name": "巴伐利亚", "weight": 0.15},
            {"name": "巴登-符腾堡", "weight": 0.10},
            {"name": "下萨克森", "weight": 0.10},
            {"name": "其它州", "weight": 0.44}
        ],
        "Japan": [
            {"name": "東京", "weight": 0.20},
            {"name": "大阪府", "weight": 0.12},
            {"name": "神奈川県", "weight": 0.10},
            {"name": "愛知県", "weight": 0.08},
            {"name": "北海道", "weight": 0.05},
            {"name": "その他", "weight": 0.45}
        ]
    },

    "country": ["China", "USA", "UK", "Germany", "Japan"],

    "user": {
        "base": 100,
        "daily_growth": 10,
        "vip_distribution": [
            {"value": 0, "weight": 0.50},
            {"value": 1, "weight": 0.20},
            {"value": 2, "weight": 0.15},
            {"value": 3, "weight": 0.10},
            {"value": 4, "weight": 0.04},
            {"value": 5, "weight": 0.01}
        ]
    },

    "product": {
        "count": 1000,
        "brand_tiers": [
            {"value": "Apple",    "weight": 0.25},
            {"value": "Samsung",  "weight": 0.20},
            {"value": "Nike",     "weight": 0.15},
            {"value": "Xiaomi",   "weight": 0.15},
            {"value": "Uniqlo",   "weight": 0.10},
            {"value": "Adidas",   "weight": 0.10},
            {"value": "Sony",     "weight": 0.05}
        ],
        "colors":    ["红色", "蓝色", "黑色", "白色", "绿色", "黄色"],
        "sizes":     ["XS", "S", "M", "L", "XL", "XXL"],
        "materials": ["棉", "聚酯纤维", "皮革", "金属", "塑料", "玻璃"],
        "category_hierarchy": [
            {
                "value": "服装", "weight": 0.40,
                "subcategories": [
                    {"value": "男装", "weight": 0.60},
                    {"value": "女装", "weight": 0.40}
                ]
            },
            {
                "value": "电子产品", "weight": 0.35,
                "subcategories": [
                    {"value": "手机", "weight": 0.50},
                    {"value": "电脑", "weight": 0.50}
                ]
            },
            {
                "value": "家居", "weight": 0.25,
                "subcategories": [
                    {"value": "家具", "weight": 0.50},
                    {"value": "装饰", "weight": 0.50}
                ]
            }
        ]
    },

    "vendor": {
        "count": 100
    },

    "discount": {
        "count": 10,
        "types": ["fixed", "coupon"]
    },

    "coupon_strategy": {
        "count": 10,
        "actions": ["popup", "push", "sms"],
        "triggers": ["view", "cart_add"]
    },

    "channel": {
        "preference_weights": [
            {"value": "mobile_app", "weight": 0.50},
            {"value": "web", "weight": 0.20},
            {"value": "mini_program", "weight": 0.10},
            {"value": "social_media", "weight": 0.10},
            {"value": "affiliate", "weight": 0.10}
        ]
    },

    "session": {
        "per_user": 2,
        "total": 1000
    },

    "event": {
        "daily_active_rate": 0.30,
        "view_distribution": {"type": "poisson", "lambda": 3},
        "cart_rate": 0.25,
        "purchase_rate": 0.15
    },

    "order": {
        "payment_methods": [
            {"value": "credit_card", "weight": 0.40},
            {"value": "third_party", "weight": 0.30},
            {"value": "prepaid_card", "weight": 0.02},
            {"value": "electronic_remittance", "weight": 0.10},
            {"value": "unionpay", "weight": 0.18}
        ],
        "purchase_channels": ["web", "mobile_app", "mini_program", "social_media", "affiliate"]
    },

    "time_range": {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31"
    }
}

# 转换时间范围并初始化全局变量
start_date = datetime.fromisoformat(CONFIG['time_range']['start_date'])
end_date = datetime.fromisoformat(CONFIG['time_range']['end_date'])
fake = Faker('zh_CN')
np.random.seed(42)
random.seed(42)


def generate_dim_user() -> pd.DataFrame:
    """
    生成用户维表数据（根据国家模拟地理分布），columns must match Hive DDL:
    """
    user_base = CONFIG['user']['base']
    days = (end_date - start_date).days
    total_users = user_base + CONFIG['user']['daily_growth'] * days

    country = CONFIG.get('default_country', 'China')
    geo_conf = CONFIG['geo'][country]  # 获取指定国家的地理配置

    geo_list = geo_conf['provinces']
    province_city_map = geo_conf.get('province_city_map', {})

    provinces = [g['name'] for g in geo_list]
    weights = [g['weight'] for g in geo_list]

    rows = []
    now = datetime.now()
    five_years_ago = now - timedelta(days=5 * 365)
    for uid in range(1, total_users + 1):
        # registration_ts in last 5 years
        reg_ts = fake.date_between(start_date=five_years_ago, end_date=now)
        # last_login_ts between reg_ts and now
        last_ts = fake.date_time_between(start_date=start_date, end_date=now)

        # 省市分布模拟
        prov = np.random.choice(provinces, p=np.array(weights) / sum(weights))
        city = random.choice(province_city_map.get(prov, ["未知"]))

        rows.append({
            "user_id": uid,
            "user_name": fake.name(),
            "gender": np.random.choice(['M', 'F', 'O'], p=[0.48, 0.50, 0.02]),
            "age": int(np.clip(norm.rvs(32, 8), 18, 65)),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "province": prov,
            "city": city,
            "registration_ts": reg_ts,
            "last_login_ts": last_ts,
            "vip_level": weighted_choice(CONFIG['user']['vip_distribution']),
            "discount_sensitivity": round(beta.rvs(1, 3)*0.7, 2),
            "return_rate": round(beta.rvs(1, 2)*0.3, 2),
            "credit_score": round(np.clip(norm.rvs(600, 50), 300, 850), 2),
            "user_channel_pref": weighted_sample(CONFIG['channel']['preference_weights'], k=2),
            "country": country
        })
    df = pd.DataFrame(rows)
    col_order = [
        "user_id", "user_name", "gender", "age",
        "email", "phone",
        "province", "city",
        "registration_ts", "last_login_ts",
        "vip_level", "discount_sensitivity", "return_rate", "credit_score",
        "user_channel_pref", "country"
    ]
    return df[col_order]
# --------------------------------
# 2. dim_product
# --------------------------------
def generate_dim_product(vendors_df: pd.DataFrame) -> pd.DataFrame:
    """
    商品维表（批量调用 GPT-2 生成 description 和 implicit_attributes）
    输出列：product_id, category, brand, vendor_id,
          unit_price, stock_qty, description, implicit_attributes
    """
    # 准备好品类层级和品牌列表
    cat_hierarchy = CONFIG['product']['category_hierarchy']
    brand_tiers   = CONFIG['product']['brand_tiers']
    product_count = CONFIG['product']['count']

    # vendor_id 列表与成本映射
    vid_list = vendors_df['vendor_id'].tolist()
    cost_map = vendors_df.set_index('vendor_id')['cost_price'].to_dict()

    # 1) 构造所有可能的 (category, brand) 组合列表
    combo_list = []
    for lvl in cat_hierarchy:
        for sub in lvl.get('subcategories', []):
            cat = f"{lvl['value']}>{sub['value']}"
            for b in brand_tiers:
                combo_list.append((cat, b['value']))

    # 2) 为每个组合生成 prompt
    prompts_desc = [
        f"Please write a detailed and engaging product description for a {cat} product by {brand}."
        for cat, brand in combo_list
    ]
    prompts_attr = [
        f"For a {cat} product by {brand}, suggest a realistic JSON of color, size, material."
        for cat, brand in combo_list
    ]

    # 3) 批量调用 GPT-2
    descriptions = batch_generate_descriptions(prompts_desc, max_new_tokens=80)
    attrs        = batch_generate_attrs(prompts_attr,   max_new_tokens=30)

    # 4) 建立快速查表
    desc_map = { combo_list[i]: descriptions[i] for i in range(len(combo_list)) }
    attr_map = { combo_list[i]: attrs[i]         for i in range(len(combo_list)) }

    # 5) 真正开始生成每条产品记录
    rows = []
    for pid in range(1, product_count + 1):
        # — 选择品类
        lvl_weights = np.array([lvl['weight'] for lvl in cat_hierarchy])
        lvl = np.random.choice(cat_hierarchy, p=lvl_weights / lvl_weights.sum())
        subs = lvl.get('subcategories', [])
        if subs:
            sub_weights = np.array([s['weight'] for s in subs])
            sub = np.random.choice(subs, p=sub_weights / sub_weights.sum())
            category = f"{lvl['value']}>{sub['value']}"
        else:
            category = lvl['value']

        # — 选择品牌
        brand = weighted_choice(brand_tiers)

        # — 供应商与价格
        vid   = random.choice(vid_list)
        cost  = cost_map[vid]
        price = round(cost * np.random.uniform(1.2, 3.0), 2)

        # — 库存
        stock = int(poisson.rvs(50))

        # — 从查表中直接读回 GPT-2 结果
        description         = desc_map[(category, brand)]
        implicit_attributes = json.dumps(attr_map[(category, brand)], ensure_ascii=False)

        rows.append({
            "product_id":          pid,
            "category":            category,
            "brand":               brand,
            "vendor_id":           vid,
            "unit_price":          price,
            "stock_qty":           stock,
            "description":         description,
            "implicit_attributes": implicit_attributes
        })

    # 6) 返回 DataFrame，并保证列顺序
    cols = ["product_id", "category", "brand", "vendor_id",
            "unit_price", "stock_qty", "description", "implicit_attributes"]
    return pd.DataFrame(rows)[cols]
# --------------------------------
# 3. dim_vendor
# --------------------------------
def generate_dim_vendor() -> pd.DataFrame:
    # 把 CONFIG['geo'] 的键（各国家）转成列表
    country_list = list(CONFIG['geo'].keys())
    rows = []
    for vid in range(1, CONFIG['vendor']['count'] + 1):
        cost = round(random.choices(
            [random.uniform(5, 1000), random.uniform(1001, 5000)],
            weights=[0.7, 0.3],
            k=1  # 只选择一个数字
        )[0], 2)
        market_price = round(cost * random.uniform(1.0125, 1.248), 2)

        # 生成退货率（贝塔分布模拟）
        return_rate = round(beta.rvs(1, 10), 2)

        # 计算综合评级
        rating = calculate_vendor_rating(
            cost_price=cost,
            market_price=market_price,
            return_rate=return_rate
        )
        rows.append({
            "vendor_id": vid,
            "vendor_name": fake.company(),
            "region": fake.province(),
            "cost_price": cost,
            "market_price": market_price,
            "vendor_rating": rating,
            "service_quality": round(random.uniform(1, 5), 2),
            "avg_delivery_time": random.randint(1, 6),
            "vendor_return_rate": round(beta.rvs(2, 8), 2),
            "distribution_areas": random.sample(["North", "South", "East", "West"], 2),
            "country": random.choice(country_list) if country_list else "China"
        })
    cols = ["vendor_id", "vendor_name", "region", "cost_price", "market_price",
            "vendor_rating", "service_quality", "avg_delivery_time",
            "vendor_return_rate", "distribution_areas", "country"]
    return pd.DataFrame(rows)[cols]


# --------------------------------
# 4. dim_discount
# --------------------------------  # 9折
#
def generate_dim_discount() -> pd.DataFrame:
    rows = []
    threshold_amount = round(random.choices(
        [random.uniform(100, 7000), random.uniform(7001, 10000)],
        weights=[0.8, 0.2],
        k=1
    )[0], 2)
    discount_value = round(max(5, min(
        random.gauss(threshold_amount * 0.9, (threshold_amount * 0.1) / 3), 500)), 2)
    for did in range(1, CONFIG['discount']['count'] + 1):
        s = fake.date_time_between(start_date=start_date, end_date=end_date)
        e = s + timedelta(days=random.randint(1, 15))
        rows.append({
            "discount_id": did,
            "discount_name": f"DISC_{did:04d}",
            "discount_type": random.choice(CONFIG['discount']['types']),
            "threshold_amount": threshold_amount,
            "discount_value": discount_value,
            "valid_from_ts": s,
            "valid_to_ts": e
        })
    cols = ["discount_id", "discount_name", "discount_type",
            "threshold_amount", "discount_value", "valid_from_ts", "valid_to_ts"]
    return pd.DataFrame(rows)[cols]


# --------------------------------
# 5. dim_time
# --------------------------------
def generate_dim_time() -> pd.DataFrame:
    rows = []
    cur = start_date
    while cur <= end_date:
        rows.append({
            "time_id": int(cur.strftime("%Y%m%d%H%M")),
            "dt": cur,
            "year": cur.year,
            "quarter": (cur.month - 1) // 3 + 1,
            "month": cur.month,
            "day": cur.day,
            "hour": cur.hour,
            "minute": cur.minute
        })
        cur += timedelta(minutes=15)
    cols = ["time_id", "dt", "year", "quarter", "month", "day", "hour", "minute"]
    return pd.DataFrame(rows)[cols]


# --------------------------------
# 6. dim_session
# --------------------------------
def generate_dim_session(user_ids: list) -> pd.DataFrame:
    rows = []
    for uid in user_ids:
        for _ in range(CONFIG['session']['per_user']):
            s = fake.date_time_between(start_date=start_date, end_date=end_date)
            e = s + timedelta(minutes=random.randint(5, 120))
            rows.append({
                "session_id": uuid.uuid4().hex[:12],
                "user_id": uid,
                "start_ts": s,
                "end_ts": e,
                "device_type": random.choice(['mobile', 'desktop', 'tablet', 'ipad', 'other']),
                "channel": random.choice(['web', 'social_media', 'search', 'ads','app','mini_program','affiliate','other']),
                "referrer": fake.url(),
                "session_date": s.date()
            })
    cols = ["session_id", "user_id", "start_ts", "end_ts",
            "device_type", "channel", "referrer", "session_date"]
    return pd.DataFrame(rows)[cols]


# --------------------------------
# 7. fact_event
# --------------------------------
def generate_behavior_events(
        users: pd.DataFrame,
        products: pd.DataFrame,
        discounts: pd.DataFrame,
        start: datetime,
        end: datetime
) -> pd.DataFrame:
    REFERRERS = ['search', 'social_media', 'ad_link', 'affiliate', 'direct']
    CAMPAIGNS = [f"camp_{i:03d}" for i in range(1, 51)]
    BROWSER_OS = [
        {"browser": "Chrome", "os": "Windows"},
        {"browser": "Safari", "os": "iOS"},
        {"browser": "Firefox", "os": "Linux"},
        {"browser": "Edge", "os": "Windows"},
        {"browser": "Chrome", "os": "Android"}
    ]

    all_ev = []
    cur = start
    while cur <= end:
        today = users.sample(frac=CONFIG['event']['daily_active_rate'])

        for _, u in today.iterrows():
            session_id = uuid.uuid4().hex[:12]
            browser_os = random.choice(BROWSER_OS)

            # VIEW行为
            vc = poisson.rvs(CONFIG['event']['view_distribution']['lambda'])
            viewed_products = products.sample(min(vc, len(products)))

            for _, p in viewed_products.iterrows():
                ts = cur + timedelta(minutes=random.randint(0, 1439))
                all_ev.append({
                    "event_id": uuid.uuid4().int & ((1 << 63) - 1),
                    "user_id": u.user_id,
                    "product_id": p.product_id,
                    "time_id": int(ts.strftime("%Y%m%d%H%M")),
                    "session_id": session_id,
                    "device_type": random.choice(['mobile', 'desktop', 'tablet', 'ipad', 'other']),
                    "ad_campaign_id": random.choice(CAMPAIGNS),
                    "event_type": "view",
                    "event_ts": ts,
                    "page_duration": random.randint(5, 80),
                    "referrer_source": random.choice(REFERRERS),
                    "discount_used": False,
                    "discount_id": None,
                    "event_attributes": browser_os,
                    "event_date": (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).date()
                })

                # CART_ADD概率
                if random.random() < 0.5:
                    ts_cart = ts + timedelta(minutes=random.randint(1, 30))
                    discount_used = random.random() < 0.5
                    discount_id = random.choice(discounts['discount_id'].tolist()) if discount_used else None

                    all_ev.append({
                        "event_id": uuid.uuid4().int & ((1 << 63) - 1),
                        "user_id": u.user_id,
                        "product_id": p.product_id,
                        "time_id": int(ts_cart.strftime("%Y%m%d%H%M")),
                        "session_id": session_id,
                        "device_type": random.choice(['mobile', 'desktop', 'tablet']),
                        "ad_campaign_id": random.choice(CAMPAIGNS),
                        "event_type": "cart_add",
                        "event_ts": ts_cart,
                        "page_duration": random.randint(5, 40),
                        "referrer_source": random.choice(REFERRERS),
                        "discount_used": discount_used,
                        "discount_id": discount_id,
                        "event_attributes": browser_os,
                        "event_date": (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).date()
                    })

                    # PURCHASE概率
                    if random.random() < 0.4:
                        ts_purchase = ts_cart + timedelta(minutes=random.randint(1, 15))
                        all_ev.append({
                            "event_id": uuid.uuid4().int & ((1 << 63) - 1),
                            "user_id": u.user_id,
                            "product_id": p.product_id,
                            "time_id": int(ts_purchase.strftime("%Y%m%d%H%M")),
                            "session_id": session_id,
                            "device_type": random.choice(['mobile', 'desktop', 'tablet']),
                            "ad_campaign_id": random.choice(CAMPAIGNS),
                            "event_type": "purchase",
                            "event_ts": ts_purchase,
                            "page_duration": random.randint(10, 60),
                            "referrer_source": random.choice(REFERRERS),
                            "discount_used": discount_used,
                            "discount_id": discount_id,
                            "event_attributes": browser_os,
                            "event_date": (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).date()
                        })

        cur += timedelta(days=1)

    cols = ["event_id", "user_id", "product_id", "time_id", "session_id",
            "device_type", "ad_campaign_id", "event_type", "event_ts",
            "page_duration", "referrer_source", "discount_used", "discount_id",
            "event_attributes", "event_date"]
    return pd.DataFrame(all_ev)[cols]


# --------------------------------
# 8. fact_order
# --------------------------------
def generate_fact_order(
        events: pd.DataFrame,
        products: pd.DataFrame,
        users: pd.DataFrame,
        vendors: pd.DataFrame,
        discounts: pd.DataFrame
) -> pd.DataFrame:
    """
    生成订单事实表，符合电商业务逻辑，输出列顺序对应 Hive DDL。
    """
    # 预生成映射加速查询
    prod_map = products.set_index("product_id").to_dict("index")
    user_map = users.set_index("user_id").to_dict("index")
    vendor_map = vendors.set_index("vendor_id").to_dict("index")
    discount_map = discounts.set_index("discount_id").to_dict("index")

    rows = []
    purchases = events[events.event_type == "purchase"]

    for ev in purchases.itertuples(index=False):
        u = user_map.get(ev.user_id)
        p = prod_map.get(ev.product_id)
        if u is None or p is None:
            continue
        v_id = p["vendor_id"]
        if v_id not in vendor_map:
            continue

        # 数量逻辑：偏向低值但允许多个
        qty = max(1, min(int(np.random.zipf(1.5)), 5))

        # 金额计算逻辑
        gross = round(p["unit_price"] * qty, 2)

        # 优惠逻辑（基础折扣 + VIP 叠加）
        base_rate = 0.04 * u["discount_sensitivity"]
        discount_amt = 0.0
        coupon_type = None
        did = getattr(ev, "discount_id", None)

        if getattr(ev, "discount_used", False):
            if u["vip_level"] >= 3:
                base_rate += 0.1 * (u["vip_level"] - 2)
            base_rate = min(base_rate, 0.3)
            discount_amt = round(gross * base_rate, 2)

            if did in discount_map:
                coupon_type = discount_map[did]["discount_type"]

        net_amt = round(gross - discount_amt, 2)

        event_day = ev.event_ts.date()
        random_hour = random.randint(0, 23)
        random_minute = random.randint(0, 59)
        random_second = random.randint(0, 59)
        ts = datetime.combine(event_day, time(random_hour, random_minute, random_second))
        rows.append({
            "order_id":         uuid.uuid4().hex,
            "user_id":          ev.user_id,
            "product_id":       ev.product_id,
            "time_id":          ev.time_id,
            "vendor_id":        v_id,
            "quantity":         qty,
            "gross_amount":     gross,
            "discount_amount":  discount_amt,
            "net_amount":       net_amt,
            "payment_method":   weighted_choice(CONFIG["order"]["payment_methods"]),
            "coupon_type":      coupon_type,
            "ad_campaign_id":   getattr(ev, "ad_campaign_id", None),
            "event_ts":         ts,
            "purchase_channel": random.choice(CONFIG["order"]["purchase_channels"]),
            "order_status":     progressive_status(ts),
            "province":         u["province"],
            "city":             u["city"],
            "order_date":       ts.date(),
            "order_year":       ts.year,
            "order_month":      ts.month,
            "order_day":        ts.day
        })

    # 定义列顺序
    cols = [
        "order_id", "user_id", "product_id", "time_id", "vendor_id",
        "quantity", "gross_amount", "discount_amount", "net_amount",
        "payment_method", "coupon_type", "ad_campaign_id", "event_ts",
        "purchase_channel", "order_status", "province", "city",
        "order_date", "order_year", "order_month", "order_day"
    ]

    df = pd.DataFrame(rows)
    df = df.reindex(columns=cols)
    return df


# --------------------------------
# 9. user_behavior_profile
# --------------------------------
def generate_user_behavior_profile(
        users: pd.DataFrame,
        orders: pd.DataFrame,
        products: pd.DataFrame
) -> pd.DataFrame:
    pcats = products.set_index("product_id")["category"].to_dict()
    rows = []

    for uid in users.user_id:
        ords = orders[orders.user_id == uid]
        rows.append({
            "user_id": uid,
            "avg_research_time": random.randint(3, 300),
            "decision_time": random.randint(5, 120),
            "purchase_intensity": round(len(ords) / 30, 2),
            "preferred_category": pcats.get(ords.product_id.mode()[0], "") if not ords.empty else "",
            "channel_preference": ords.purchase_channel.unique().tolist(),
            "social_influence_factor": round(beta.rvs(2, 5), 2),
            "profile_date": (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).date()
        })
    cols = ["user_id", "avg_research_time", "decision_time", "purchase_intensity",
            "preferred_category", "channel_preference", "social_influence_factor", "profile_date"]
    return pd.DataFrame(rows)[cols]


# --------------------------------
# 10. coupon_strategy
# --------------------------------
def generate_coupon_strategy(
        users: pd.DataFrame
    ) -> pd.DataFrame:
    rows = []

    for sid in range(1, CONFIG['coupon_strategy']['count'] + 1):
        rows.append({
            "strategy_id": sid,
            "min_discount_sensitivity": round(random.uniform(0.01, 0.42), 2),
            "min_view_count": random.randint(2, 10),
            "min_cart_count": random.randint(1, 5),
            "trigger_type": random.choice(CONFIG['coupon_strategy']['triggers']),
            "action_type": random.choice(CONFIG['coupon_strategy']['actions']),
            "coupon_id": random.randint(1, CONFIG['discount']['count'])
        })
    cols = ["strategy_id", "min_discount_sensitivity", "min_view_count",
            "min_cart_count", "trigger_type", "action_type", "coupon_id"]
    return pd.DataFrame(rows)[cols]
if __name__ == "__main__":
    # 生成维度数据
    dim_vendor = generate_dim_vendor()  # 先生成供应商
    dim_user = generate_dim_user()
    dim_product = generate_dim_product(dim_vendor)
    dim_discount = generate_dim_discount()
    dim_time = generate_dim_time()
    dim_session = generate_dim_session(dim_user.user_id.tolist())

    # after you’ve built dim_user, dim_product, dim_session, dim_discount…
    df_events = generate_behavior_events(
        users=dim_user,
        products=dim_product,
        discounts=dim_discount,
        start=start_date,
        end=end_date
    )

    # 生成订单数据
    df_orders = generate_fact_order(
        events=df_events,
        products=dim_product,
        users=dim_user,
        vendors=dim_vendor,
        discounts=dim_discount
    )

    # 修复变量引用
    fact_event = df_events  # 统一变量命名
    fact_order = df_orders

    # 修复用户画像生成调用
    df_profiles = generate_user_behavior_profile(
        users=dim_user,
        orders=df_orders,  # 确保包含purchase_channel字段
        products=dim_product
    )

    # 生成辅助表
    user_behavior_profile = generate_user_behavior_profile(dim_user, fact_order, dim_product)
    coupon_strategy = generate_coupon_strategy(users=dim_user)