# ================================================
# 完整且优化的数据生成脚本
# Filename: generate_ecommerce_data.py
# ================================================
import torch
import pandas as pd
import numpy as np
from faker import Faker
import uuid
import json
import random
from datetime import datetime, timedelta, time
from scipy.stats import beta, poisson, norm, zipf
import re

# ================================================
# 全局配置加载
# ================================================
CONFIG = {
    # 各国主力市场省/州/地区分布及权重
    "geo": {
        "China": {
            "provinces": [
                {"name": "广东", "weight": 0.18}, {"name": "江苏", "weight": 0.12},
                {"name": "浙江", "weight": 0.10}, {"name": "山东", "weight": 0.08},
                {"name": "河南", "weight": 0.06}, {"name": "其它省份", "weight": 0.46}
            ],
            "province_city_map": {
                "广东": ["广州", "深圳", "佛山", "东莞"], "江苏": ["南京", "苏州", "无锡", "常州"],
                "浙江": ["杭州", "宁波", "温州", "绍兴"], "山东": ["青岛", "济南"], "河南": ["郑州"]
            }
        },
        "USA": {
            "provinces": [
                {"name": "California", "weight": 0.15}, {"name": "Texas", "weight": 0.12},
                {"name": "New York", "weight": 0.10}, {"name": "Florida", "weight": 0.08},
                {"name": "Illinois", "weight": 0.05}, {"name": "Other States", "weight": 0.50}
            ],
            "province_city_map": {
                "California": ["Los Angeles", "San Francisco", "San Diego"],
                "Texas": ["Houston", "Dallas", "Austin"],
                "New York": ["New York City", "Buffalo", "Rochester"]
            }
        }
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
        "base": 1000,
        "daily_growth": 50,
        "vip_distribution": [
            {"value": 0, "weight": 0.50}, {"value": 1, "weight": 0.20},
            {"value": 2, "weight": 0.15}, {"value": 3, "weight": 0.10},
            {"value": 4, "weight": 0.04}, {"value": 5, "weight": 0.01}
        ]
    },
    "product": {
        "count": 1000,
        "brand_tiers": [
            {"value": "Apple", "weight": 0.25}, {"value": "Samsung", "weight": 0.20},
            {"value": "Nike", "weight": 0.15}, {"value": "Xiaomi", "weight": 0.15},
            {"value": "Uniqlo", "weight": 0.10}, {"value": "Adidas", "weight": 0.10},
            {"value": "Sony", "weight": 0.05}
        ],
        "category_hierarchy": [
            {"value": "服装", "weight": 0.40, "subcategories": [{"value": "男装", "weight": 0.60}, {"value": "女装", "weight": 0.40}]},
            {"value": "电子产品", "weight": 0.35, "subcategories": [{"value": "手机", "weight": 0.50}, {"value": "电脑", "weight": 0.50}]},
            {"value": "家居", "weight": 0.25, "subcategories": [{"value": "家具", "weight": 0.50}, {"value": "装饰", "weight": 0.50}]}
        ]
    },
    "vendor": {"count": 100},
    "discount": {"count": 20, "types": ["fixed", "coupon", "percentage"]},
    "session": {"per_user": 2},
    "event": {
        "daily_active_rate": 0.30,
        "view_distribution": {"type": "poisson", "lambda": 3},
        "cart_add_rate": 0.5, # 浏览后加入购物车的概率
        "purchase_rate": 0.4  # 加入购物车后购买的概率
    },
    "order": {
        "payment_methods": [
            {"value": "credit_card", "weight": 0.40}, {"value": "third_party", "weight": 0.30},
            {"value": "unionpay", "weight": 0.28}, {"value": "prepaid_card", "weight": 0.02}
        ],
        "purchase_channels": ["web", "mobile_app", "mini_program", "social_media", "affiliate"]
    },
    "time_range": {"start_date": "2025-01-01", "end_date": "2025-06-30"}
}

# 初始化全局变量
start_date = datetime.fromisoformat(CONFIG['time_range']['start_date'])
end_date = datetime.fromisoformat(CONFIG['time_range']['end_date'])
fake_cn = Faker('zh_CN')
fake_en = Faker('en_US')
np.random.seed(42)
random.seed(42)

# ================================================
# 辅助函数 (补全)
# ================================================
def weighted_choice(items):
    """根据权重选择单个项目"""
    values = [item['value'] for item in items]
    weights = np.array([item['weight'] for item in items])
    return np.random.choice(values, p=weights / weights.sum())

def weighted_sample(items, k):
    """根据权重不重复抽样k个项目"""
    values = [item['value'] for item in items]
    weights = np.array([item['weight'] for item in items])
    return np.random.choice(values, size=k, replace=False, p=weights / weights.sum()).tolist()

def calculate_vendor_rating(cost_price, market_price, return_rate, service_quality, delivery_time):
    """计算供应商综合评级"""
    price_advantage = (market_price - cost_price) / cost_price if cost_price > 0 else 0
    # 归一化和加权
    score = (0.4 * min(price_advantage, 1)) + (0.3 * (1 - return_rate)) + (0.2 * (service_quality / 5)) + (0.1 * (1 - (delivery_time / 10)))
    return round(1 + score * 4, 2) # 映射到1-5分

def progressive_status(order_ts: datetime) -> str:
    """根据订单时间模拟一个真实的订单状态"""
    now = datetime.now()
    delta_days = (now - order_ts).days
    if delta_days < 0: return "Pending"
    if delta_days < 1: return "Processing"
    if delta_days < 3: return "Shipped"
    if delta_days < 7: return "Delivered"
    if random.random() < 0.05: # 5%的概率退货
        return "Returned"
    return "Completed"

# ================================================
# 维度表生成函数
# ================================================

def generate_dim_user() -> pd.DataFrame:
    user_base = CONFIG['user']['base']
    days = (end_date - start_date).days
    total_users = user_base + CONFIG['user']['daily_growth'] * days
    rows = []
    for uid in range(1, total_users + 1):
        country = random.choice(CONFIG['country'])
        geo_conf = CONFIG['geo'][country]
        fake = fake_cn if country == 'China' else fake_en
        
        provinces = [g['name'] for g in geo_conf['provinces']]
        weights = [g['weight'] for g in geo_conf['provinces']]
        prov = np.random.choice(provinces, p=np.array(weights) / sum(weights))
        city = random.choice(geo_conf.get('province_city_map', {}).get(prov, ["Unknown"]))

        reg_ts = fake.date_time_between(start_date=start_date - timedelta(days=365*2), end_date=end_date)
        last_ts = fake.date_time_between(start_date=reg_ts, end_date=end_date)

        rows.append({
            "user_id": uid,
            "user_name": fake.name(),
            "gender": np.random.choice(['M', 'F', 'O'], p=[0.48, 0.50, 0.02]),
            "age": int(np.clip(norm.rvs(32, 8), 18, 65)),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "country": country,
            "province": prov,
            "city": city,
            "registration_date": reg_ts.date(),
            "last_login": last_ts,
            "vip_level": weighted_choice(CONFIG['user']['vip_distribution']),
            "discount_sensitivity": round(beta.rvs(1, 3) * 0.7, 4),
            "return_rate": round(beta.rvs(1, 10) * 0.3, 4),
            "credit_score": round(np.clip(norm.rvs(650, 80), 300, 850))
        })
    return pd.DataFrame(rows)

def generate_dim_product(vendors_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    vid_list = vendors_df['vendor_id'].tolist()
    cost_map = vendors_df.set_index('vendor_id')['cost_price'].to_dict()

    for pid in range(1, CONFIG['product']['count'] + 1):
        # 选择品类
        cat_hier = CONFIG['product']['category_hierarchy']
        lvl_weights = np.array([lvl['weight'] for lvl in cat_hier])
        lvl = np.random.choice(cat_hier, p=lvl_weights / lvl_weights.sum())
        subs = lvl.get('subcategories', [])
        if subs:
            sub_weights = np.array([s['weight'] for s in subs])
            sub = np.random.choice(subs, p=sub_weights / sub_weights.sum())
            category = f"{lvl['value']}>{sub['value']}"
        else:
            category = lvl['value']

        brand = weighted_choice(CONFIG['product']['brand_tiers'])
        vid = random.choice(vid_list)
        cost = cost_map[vid]
        price = round(cost * np.random.uniform(1.3, 3.5), 2)
        stock = int(poisson.rvs(150))
        
        # 简化的描述和属性生成
        desc = f"这是一款由{brand}品牌制造的优质{category.split('>')[-1]}，设计独特，品质保证。"
        attrs = {"color": random.choice(["红", "黑", "白"]), "size": random.choice(["M", "L", "XL"])}

        rows.append({
            "product_id": pid,
            "category": category,
            "brand": brand,
            "unit_price": price,
            "stock": stock,
            "description": desc,
            "implicit_attributes": json.dumps(attrs, ensure_ascii=False),
            "vendor_id": vid
        })
    return pd.DataFrame(rows)

def generate_dim_vendor() -> pd.DataFrame:
    rows = []
    for vid in range(1, CONFIG['vendor']['count'] + 1):
        country = random.choice(CONFIG['country'])
        fake = fake_cn if country == 'China' else fake_en
        cost = round(random.uniform(10, 800), 2)
        market_price = round(cost * random.uniform(1.05, 1.25), 2)
        return_rate = round(beta.rvs(1, 15), 4)
        service_quality = round(random.uniform(2.5, 5), 2)
        avg_delivery_time = random.randint(1, 8)
        rating = calculate_vendor_rating(cost, market_price, return_rate, service_quality, avg_delivery_time)

        rows.append({
            "vendor_id": vid,
            "vendor_name": fake.company(),
            "country": country,
            "cost_price": cost,
            "market_price": market_price,
            "vendor_rating": rating,
            "service_quality": service_quality,
            "avg_delivery_time": avg_delivery_time,
            "vendor_return_rate": return_rate,
            "distribution_areas": random.sample(["华东", "华南", "华北", "华中", "西南", "西北"], k=random.randint(1,3))
        })
    return pd.DataFrame(rows)

def generate_dim_discount() -> pd.DataFrame:
    rows = []
    for did in range(1, CONFIG['discount']['count'] + 1):
        s = fake_cn.date_time_between(start_date=start_date, end_date=end_date)
        e = s + timedelta(days=random.randint(7, 45))
        dtype = random.choice(CONFIG['discount']['types'])
        value = round(random.uniform(0.75, 0.95), 2) if dtype == 'percentage' else round(random.uniform(5, 50), 2)
        
        rows.append({
            "discount_id": did,
            "discount_name": f"D_{dtype.upper()}_{uuid.uuid4().hex[:4]}",
            "discount_type": dtype,
            "threshold_amount": round(random.choice([0, 100, 200, 500]) * random.uniform(0.9, 1.1), 2),
            "discount_value": value,
            "valid_from": s,
            "valid_to": e
        })
    return pd.DataFrame(rows)

def generate_dim_time() -> pd.DataFrame:
    rows = []
    cur = start_date
    while cur <= end_date:
        rows.append({
            "time_id": int(cur.strftime("%Y%m%d")),
            "dt": cur.date(),
            "year": cur.year,
            "quarter": (cur.month - 1) // 3 + 1,
            "month": cur.month,
            "day": cur.day,
            "hour": cur.hour,
            "minute": cur.minute,
            "day_of_week": cur.weekday() + 1, # Monday=1
            "is_weekend": 1 if cur.weekday() >= 5 else 0
        })
        cur += timedelta(days=1)
    return pd.DataFrame(rows).drop_duplicates(subset=['time_id'])

# ================================================
# 事实表生成函数
# ================================================

def generate_fact_event(users: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    all_events = []
    total_days = (end_date - start_date).days
    user_ids = users['user_id'].tolist()
    product_ids = products['product_id'].tolist()
    
    for day_offset in range(total_days):
        current_date = start_date + timedelta(days=day_offset)
        # 每日活跃用户
        active_users_count = int(len(user_ids) * CONFIG['event']['daily_active_rate'] * (random.uniform(0.8, 1.2)))
        active_user_ids = np.random.choice(user_ids, size=active_users_count, replace=False)
        
        for user_id in active_user_ids:
            session_id = uuid.uuid4().hex
            # 模拟浏览事件
            view_count = poisson.rvs(CONFIG['event']['view_distribution']['lambda'])
            viewed_product_ids = np.random.choice(product_ids, size=min(view_count, len(product_ids)), replace=False)
            
            cart_added_products = []
            
            for product_id in viewed_product_ids:
                event_ts = current_date + timedelta(seconds=random.randint(0, 86399))
                all_events.append({
                    "event_id": uuid.uuid4().hex, "user_id": user_id, "product_id": product_id,
                    "session_id": session_id, "event_type": "view", "event_ts": event_ts,
                    "event_date": event_ts.date()
                })
                
                # 模拟加购事件
                if random.random() < CONFIG['event']['cart_add_rate']:
                    cart_ts = event_ts + timedelta(seconds=random.randint(10, 300))
                    all_events.append({
                       "event_id": uuid.uuid4().hex, "user_id": user_id, "product_id": product_id,
                       "session_id": session_id, "event_type": "cart_add", "event_ts": cart_ts,
                       "event_date": cart_ts.date()
                    })
                    cart_added_products.append(product_id)

            # 模拟购买事件 (基于购物车中的商品)
            for product_id in cart_added_products:
                 if random.random() < CONFIG['event']['purchase_rate']:
                    purchase_ts = all_events[-1]['event_ts'] + timedelta(seconds=random.randint(60, 1800))
                    all_events.append({
                       "event_id": uuid.uuid4().hex, "user_id": user_id, "product_id": product_id,
                       "session_id": session_id, "event_type": "purchase", "event_ts": purchase_ts,
                       "event_date": purchase_ts.date()
                    })
                    
    return pd.DataFrame(all_events)

def generate_fact_order(events: pd.DataFrame, products: pd.DataFrame, users: pd.DataFrame, discounts: pd.DataFrame) -> pd.DataFrame:
    purchases = events[events['event_type'] == 'purchase'].copy()
    if purchases.empty:
        return pd.DataFrame()
        
    prod_map = products.set_index("product_id").to_dict("index")
    user_map = users.set_index("user_id").to_dict("index")
    
    rows = []
    for _, ev in purchases.iterrows():
        user = user_map.get(ev.user_id)
        prod = prod_map.get(ev.product_id)
        if not user or not prod:
            continue
            
        qty = max(1, int(zipf.rvs(2.5)))
        gross = round(prod["unit_price"] * qty, 2)
        
        # 折扣逻辑
        discount_amt = 0.0
        did = None
        # 随机应用一个有效折扣
        if random.random() < user['discount_sensitivity']:
             valid_discounts = discounts[discounts['valid_from'] <= ev.event_ts]
             valid_discounts = valid_discounts[valid_discounts['valid_to'] >= ev.event_ts]
             valid_discounts = valid_discounts[valid_discounts['threshold_amount'] <= gross]
             if not valid_discounts.empty:
                 chosen_discount = valid_discounts.sample(1).iloc[0]
                 did = chosen_discount['discount_id']
                 if chosen_discount['discount_type'] == 'fixed':
                     discount_amt = chosen_discount['discount_value']
                 elif chosen_discount['discount_type'] == 'coupon':
                     discount_amt = chosen_discount['discount_value']
                 elif chosen_discount['discount_type'] == 'percentage':
                     discount_amt = gross * chosen_discount['discount_value']

        net_amt = round(gross - discount_amt, 2)
        
        ts = ev.event_ts
        rows.append({
            "order_id": uuid.uuid4().hex,
            "user_id": ev.user_id,
            "product_id": ev.product_id,
            "discount_id": did,
            "time_id": int(ts.strftime("%Y%m%d")),
            "vendor_id": prod['vendor_id'],
            "quantity": qty,
            "gross_amount": gross,
            "discount_amount": round(discount_amt, 2),
            "net_amount": max(0, net_amt),
            "payment_method": weighted_choice(CONFIG["order"]["payment_methods"]),
            "purchase_channel": random.choice(CONFIG["order"]["purchase_channels"]),
            "order_status": progressive_status(ts),
            "province": user["province"],
            "city": user["city"],
            "order_date": ts.date()
        })
        
    return pd.DataFrame(rows)

# ================================================
# 主程序
# ================================================
if __name__ == "__main__":
    print("开始生成维度表...")
    dim_vendor = generate_dim_vendor()
    dim_user = generate_dim_user()
    dim_product = generate_dim_product(dim_vendor)
    dim_discount = generate_dim_discount()
    dim_time = generate_dim_time()
    
    print(f"  - dim_vendor: {len(dim_vendor)} 条")
    print(f"  - dim_user: {len(dim_user)} 条")
    print(f"  - dim_product: {len(dim_product)} 条")
    print(f"  - dim_discount: {len(dim_discount)} 条")
    print(f"  - dim_time: {len(dim_time)} 条")
    
    # 保存为CSV（上传HDFS的准备）
    # 为了在Hive中正确处理中文字符和复杂数据类型，建议保存为UTF-8编码的CSV，或直接使用Spark写入Parquet/ORC
    output_dir = "./ecommerce_data"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dim_vendor.to_csv(f"{output_dir}/dim_vendor.csv", index=False, encoding='utf-8')
    dim_user.to_csv(f"{output_dir}/dim_user.csv", index=False, encoding='utf-8')
    dim_product.to_csv(f"{output_dir}/dim_product.csv", index=False, encoding='utf-8')
    dim_discount.to_csv(f"{output_dir}/dim_discount.csv", index=False, encoding='utf-8')
    dim_time.to_csv(f"{output_dir}/dim_time.csv", index=False, encoding='utf-8')
    
    print("\n开始生成事实表...")
    fact_event = generate_fact_event(dim_user, dim_product)
    fact_order = generate_fact_order(fact_event, dim_product, dim_user, dim_discount)

    print(f"  - fact_event: {len(fact_event)} 条")
    print(f"  - fact_order: {len(fact_order)} 条")

    fact_event.to_csv(f"{output_dir}/fact_event.csv", index=False, encoding='utf-8')
    fact_order.to_csv(f"{output_dir}/fact_order.csv", index=False, encoding='utf-8')
    
    print(f"\n数据生成完毕，已保存至 '{output_dir}' 目录。")
    print("下一步：将CSV文件上传到HDFS，并使用下面的Hive DDL建表和加载数据。")