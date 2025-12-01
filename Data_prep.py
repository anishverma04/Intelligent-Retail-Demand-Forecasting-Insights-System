# Create synthetic data first (or use Kaggle retail datasets)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2024-11-30', freq='D')
n_stores = 5
n_products = 20

data = []
for date in dates:
    for store in range(n_stores):
        for product in range(n_products):
            base_demand = np.random.poisson(50)
            seasonal = 20 * np.sin(2 * np.pi * date.dayofyear / 365)
            promo_effect = np.random.choice([0, 30], p=[0.8, 0.2])
            
            sales = max(0, base_demand + seasonal + promo_effect + np.random.normal(0, 10))
            
            data.append({
                'date': date,
                'store_id': store,
                'product_id': product,
                'sales': sales,
                'price': 10 + np.random.uniform(-2, 2),
                'promotion': 1 if promo_effect > 0 else 0,
                'inventory': sales + np.random.randint(10, 50),
                'day_of_week': date.dayofweek,
                'month': date.month,
                'is_holiday': 1 if date.dayofweek >= 5 else 0
            })

df = pd.DataFrame(data)
df.to_csv('retail_sales.csv', index=False)
