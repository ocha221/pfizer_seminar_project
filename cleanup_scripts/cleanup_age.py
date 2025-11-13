import pandas as pd
import os

df = pd.read_csv('filtered_dataset/regression/dataset_with_inferred.csv')

MODE = 'fill'  # drop/zero

columns_to_check = ['customer_id', 'age', 'gender', 'income', 'purchase_date', 
                    'product_id', 'quantity', 'spend_amount', 'payment_method', 
                    'is_returning', 'lat', 'lon']

df = df.drop_duplicates(subset=columns_to_check, keep='first')

def is_erroneous(value):
    return pd.isna(value) or value == 9999999.0 

df['is_erroneous'] = df[['income', 'age', 'lat', 'lon']].apply(lambda x: x.apply(is_erroneous)).any(axis=1)
## will drop erroneous income rows
if MODE == 'drop':
    print(f"Original rows: {len(df)}")
    df = df[~df['is_erroneous']].copy()
    df = df.drop(columns=['is_erroneous'])
    print(f"Rows after removing erroneous income entries: {len(df)}")
elif MODE == 'zero':
    df['income'] = df['income'].where(~df['is_erroneous'], 0)
    df = df.drop(columns=['is_erroneous'])
elif MODE == 'fill':
    median_income = df.loc[~df['is_erroneous'], 'income'].median()
    df['income'] = df['income'].where(~df['is_erroneous'], median_income)
    median_age = df.loc[~df['is_erroneous'], 'age'].median()
    median_lat = df.loc[~df['is_erroneous'], 'lat'].median()
    median_lon = df.loc[~df['is_erroneous'], 'lon'].median()
    df['age'] = df['age'].where(~df['is_erroneous'], median_age)
    ## df['lat'] = df['lat'].where(~df['is_erroneous'], median_lat)
    ## df['lon'] = df['lon'].where(~df['is_erroneous'], median_lon)
    df = df.drop(columns=['is_erroneous'])

os.makedirs('filtered_dataset/classification/test', exist_ok=True)
df.to_csv(f'filtered_dataset/classification/test/income_{MODE}_file.csv', index=False)
