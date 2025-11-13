import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Clean and infer missing price data')
parser.add_argument('--in-place', action='store_true', 
                    help='Process in-place mode: reads from and writes to the same file with inferred values')
parser.add_argument('--input', type=str, default='filtered_dataset/raw/final.csv',
                    help='Input CSV file path (default: filtered_dataset/raw/final.csv)')
parser.add_argument('--output', type=str, default=None,
                    help='Output CSV file path (only used in in-place mode)')
args = parser.parse_args()

df = pd.read_csv(args.input)

columns_to_check = ['customer_id', 'age', 'gender', 'income', 'purchase_date', 
                    'product_id', 'quantity', 'spend_amount', 'payment_method', 
                    'is_returning', 'lat', 'lon']

print("=" * 80)
print("DUPLICATE REMOVAL")
print("=" * 80)
print(f"Original rows: {len(df)}")

df = df.drop_duplicates(subset=columns_to_check, keep='first')

print(f"Rows after removing duplicates: {len(df)}")
print(f"Duplicates removed: {len(df) - len(df)}")
print("\n")

def is_erroneous(value):
    return pd.isna(value) or value == 999999.0


df['is_erroneous'] = df['spend_amount'].apply(is_erroneous)
df['quantity_sign'] = df['quantity'].apply(lambda x: 'positive' if x > 0 else 'negative')


valid_data = df[~df['is_erroneous']].copy()


summary = valid_data.groupby(['product_id', 'quantity_sign']).agg({
    'spend_amount': ['median', 'mean', 'count'],
    'quantity': ['median', 'mean']
}).round(2)

print("=" * 80)
print("SUMMARY: Median Spend by Product and Quantity Sign")
print("=" * 80)
print(summary)
print("\n")


valid_data['unit_price'] = valid_data['spend_amount'] / valid_data['quantity'].abs()

unit_price_summary = valid_data.groupby(['product_id', 'quantity_sign']).agg({
    'unit_price': ['median', 'mean', 'count', 'std']
}).round(2)

print("=" * 80)
print("UNIT PRICE ANALYSIS (Spend per Item)")
print("=" * 80)
print(unit_price_summary)
print("\n")


erroneous_entries = df[df['is_erroneous']]
print("=" * 80)
print(f"ERRONEOUS ENTRIES: {len(erroneous_entries)} total")
print("=" * 80)
print(erroneous_entries[['customer_id', 'product_id', 'quantity', 'spend_amount', 'quantity_sign']].head(20))
print("\n")

median_unit_prices = valid_data.groupby(['product_id', 'quantity_sign'])['unit_price'].median().reset_index()
median_unit_prices.columns = ['product_id', 'quantity_sign', 'median_unit_price']

df = df.merge(median_unit_prices, on=['product_id', 'quantity_sign'], how='left')

df['spend_amount_cleaned'] = df['spend_amount'].copy()
mask = df['is_erroneous']
df.loc[mask, 'spend_amount_cleaned'] = df.loc[mask, 'median_unit_price'] * df.loc[mask, 'quantity'].abs()

df['inferred'] = mask & df['spend_amount_cleaned'].notna()

print("=" * 80)
print("BEFORE/AFTER COMPARISON (First 20 cleaned entries)")
print("=" * 80)
cleaned_entries = df[df['inferred']].copy()
cleaned_entries = cleaned_entries[['product_id', 'quantity', 'spend_amount', 'spend_amount_cleaned', 'median_unit_price']].head(20)
print(cleaned_entries)
print("\n")

still_missing = df['spend_amount_cleaned'].isna().sum()
cleaned_count = df['inferred'].sum()

print("=" * 80)
print("CLEANING SUMMARY")
print("=" * 80)
print(f"Total erroneous entries: {df['is_erroneous'].sum()}")
print(f"Successfully cleaned: {cleaned_count}")
print(f"Still missing (no reference data): {still_missing}")
print("\n")

if args.in_place:
    df_output = df.copy()
    df_output['spend_amount'] = df_output['spend_amount_cleaned']
    df_output = df_output.drop(['is_erroneous', 'quantity_sign', 'spend_amount_cleaned', 'median_unit_price'], axis=1)
    
    output_file = args.output if args.output else args.input
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_output.to_csv(output_file, index=False)
    
    print("=" * 80)
    print("OUTPUT SUMMARY (IN-PLACE MODE)")
    print("=" * 80)
    print(f"Dataset saved to: {output_file}")
    print(f"  - Total rows: {len(df_output)}")
    print(f"  - Rows with inferred values: {cleaned_count}")
else:
    df_with_inferred = df.copy()
    df_with_inferred['spend_amount'] = df_with_inferred['spend_amount_cleaned']
    df_with_inferred = df_with_inferred.drop(['is_erroneous', 'quantity_sign', 'spend_amount_cleaned', 'median_unit_price'], axis=1)

    df_raw_only = df[~df['is_erroneous']].copy()
    df_raw_only = df_raw_only.drop(['is_erroneous', 'quantity_sign', 'spend_amount_cleaned', 'median_unit_price', 'inferred'], axis=1)

    output_filename_inferred = 'filtered_dataset/regression/dataset_with_inferred.csv'
    output_filename_raw = 'filtered_dataset/regression/dataset_filtered_raw.csv'

    os.makedirs(os.path.dirname(output_filename_inferred), exist_ok=True)

    df_with_inferred.to_csv(output_filename_inferred, index=False)
    df_raw_only.to_csv(output_filename_raw, index=False)

    print("=" * 80)
    print("OUTPUT SUMMARY")
    print("=" * 80)
    print(f"Dataset with inferred values saved to: {output_filename_inferred}")
    print(f"  - Total rows: {len(df_with_inferred)}")
    print(f"  - Rows with inferred values: {cleaned_count}")
    print(f"\nDataset with raw data only saved to: {output_filename_raw}")
    print(f"  - Total rows: {len(df_raw_only)}")
    print(f"  - All rows have original, non-erroneous data")