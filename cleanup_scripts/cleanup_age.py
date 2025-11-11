import numpy as np
import pandas as pd
import os

df = pd.read_csv('/Users/chai/pfizer_seminar/filtered_dataset/regression/dataset_with_inferred.csv')

columns_to_check = ['customer_id', 'age', 'gender', 'income', 'purchase_date', 
                    'product_id', 'quantity', 'spend_amount', 'payment_method', 
                    'is_returning', 'lat', 'lon']

df = df.drop_duplicates(subset=columns_to_check, keep='first')

def is_erroneous(value):
    return pd.isna(value) or value == 9999999.0

df['is_erroneous'] = df['income'].apply(is_erroneous)

## will drop erroneous income rows
df = df[~df['is_erroneous']].copy()
df = df.drop(columns=['is_erroneous'])
print(f"Rows after removing erroneous income entries: {len(df)}")
os.makedirs('filtered_dataset/var', exist_ok=True)
df.to_csv('filtered_dataset/var/no_erroneous_income_imputed_spend.csv', index=False)
