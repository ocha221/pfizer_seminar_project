import pandas as pd
import numpy as np
import os 

COLUMNS_TO_REMOVE = []  

DEFAULT_VALUES = {
    # 'lat': 0.0,
    # 'lon': 0.0,
    # 'income': 50000,
    'gender': 'Other',
}

REQUIRED_COLUMNS = [
    
] # AN dn exei value == drop row

CAP_OUTLIERS = {
    'income': False,      # (900,81,999999) => (900,81,900)
    'age': False,        
    'quantity': False,
}

ALLOWED_VALUES = {
    'gender': ['Male', 'Female', 'Other'],
    'payment_method': ['Cash', 'Card', 'Online Wallet', 'Crypt0'],
}

FORCE_INVALID_TO_DEFAULT = {
    'gender': 'Other',           # unknown => 'Other'
    #....
}


DATE_FORMAT = 'mixed'  #  'mixed'


def clean_csv_file(input_file, output_file, quarter=None):
    df = pd.read_csv(input_file, sep=',', header=None)

    columns = [
        'customer_id', 'age', 'gender', 'income', 'purchase_date', 
        'product_id', 'quantity', 'spend_amount', 'payment_method', 
        'is_returning', 'lat', 'lon'
    ]
    df_clean = df.iloc[1:].copy()
    
    while len(df_clean.columns) < len(columns):
        df_clean[f'Extra_{len(df_clean.columns)}'] = np.nan
    
    df_clean.columns = columns[:len(df_clean.columns)]
    
    print(f"Initial rows: {len(df_clean)}")
    
    if COLUMNS_TO_REMOVE:
        df_clean = df_clean.drop(columns=COLUMNS_TO_REMOVE, errors='ignore')
    
    numeric_columns = ['age', 'income', 'quantity', 'spend_amount', 'lat', 'lon']
    numeric_columns = [col for col in numeric_columns if col in df_clean.columns]
    
    for col in numeric_columns:
        df_clean[col] = df_clean[col].replace('', np.nan)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # dates
    if 'purchase_date' in df_clean.columns:
        df_clean['purchase_date'] = df_clean['purchase_date'].replace('', np.nan)
        
        
        df_clean['purchase_date'] = pd.to_datetime(df_clean['purchase_date'], format=DATE_FORMAT, errors='coerce')
        
        invalid_dates = df_clean['purchase_date'].isna().sum()
        if invalid_dates > 0:
            print(f"Found {invalid_dates} invalid dates")
    
    for col, allowed in ALLOWED_VALUES.items():
        if col in df_clean.columns:
            before = len(df_clean)

            if col in FORCE_INVALID_TO_DEFAULT:
                invalid_mask = ~df_clean[col].isin(allowed)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    df_clean.loc[invalid_mask, col] = FORCE_INVALID_TO_DEFAULT[col]
                    print(f"Forced {invalid_count} invalid '{col}' values to '{FORCE_INVALID_TO_DEFAULT[col]}'")
            else:
                df_clean = df_clean[df_clean[col].isin(allowed)]
                removed = before - len(df_clean)
                if removed > 0:
                    print(f"Removed {removed} rows with invalid '{col}' values")
    
    for col, default in DEFAULT_VALUES.items():
        if col in df_clean.columns:
            filled = df_clean[col].isna().sum()
            df_clean[col] = df_clean[col].fillna(default)
            if filled > 0:
                print(f"Filled {filled} missing values in '{col}' with {default}")
    
    for col in REQUIRED_COLUMNS:
        if col in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean[col].notna()]
            removed = before - len(df_clean)
            if removed > 0:
                print(f"Removed {removed} rows with missing '{col}'")

    text_columns = ['gender', 'payment_method', 'is_returning']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace('nan', np.nan)
    
    
    if 'is_returning' in df_clean.columns:
        df_clean['is_returning'] = df_clean['is_returning'].map({
            'yes': True, 'Yes': True, 'YES': True, 'True': True, 'true': True,
            'no': False, 'No': False, 'NO': False, 'False': False, 'false': False
        }).fillna(False)
    
    if 'payment_method' in df_clean.columns:
        df_clean['payment_method'] = df_clean['payment_method'].map({
            'Crypt0': 'Crypto',
            'Cash': 'Cash', 'Card': 'Card', 'Online Wallet': 'Online Wallet'
        })
    
    if quarter is not None:
        df_clean['is_q1'] = (quarter == 1)
        df_clean['is_q2'] = (quarter == 2)
        df_clean['is_q3'] = (quarter == 3)
    
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Final rows: {len(df_clean)}")
    print(f"Rows removed: {len(df) - 1 - len(df_clean)}")
    
    df_clean.to_csv(output_file, index=False, encoding='utf-8')
    
    return df_clean

def analyze_data(df):
    """Print basic statistics about the cleaned data"""
    print("\n=== Data Analysis ===")
    print(f"Total rows: {len(df)}")
    print(f"\nMissing values per column:")
    print(df.isna().sum())
    print(f"\nNumeric column statistics:")
    print(df.describe())
    
    if 'gender' in df.columns:
        print(f"\nGender distribution:")
        print(df['gender'].value_counts())
    
    if 'payment_method' in df.columns:
        print(f"\nPayment method distribution:")
        print(df['payment_method'].value_counts())

if __name__ == "__main__":
    input_files = [
        "filtered_dataset/raw/Q1.csv",
        "filtered_dataset/raw/Q2.csv",
        "filtered_dataset/raw/Q3.csv"
    ]
    output_files = ["Q1_cleaned.csv", "Q2_cleaned.csv", "Q3_cleaned.csv"]
    
    cleaned_dfs = []
    for i, (input_file, output_file) in enumerate(zip(input_files, output_files), start=1):
        try:
            print(f"\n{'='*50}")
            print(f"Processing Q{i}")
            print('='*50)
            cleaned_df = clean_csv_file(input_file, output_file, quarter=i)
            analyze_data(cleaned_df)
            cleaned_dfs.append(cleaned_df)
            print(f"\nCleaned data saved to: {output_file}")
        except Exception as e:
            print(f"Error cleaning CSV file: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if cleaned_dfs:
        print(f"\n{'='*50}")
        print("Merging all cleaned files into final.csv")
        print('='*50)
        final_df = pd.concat(cleaned_dfs, ignore_index=True)
        os.makedirs("filtered_dataset/raw", exist_ok=True)
        final_df.to_csv("filtered_dataset/raw/final.csv", index=False, encoding='utf-8')
        print(f"Final merged data saved with {len(final_df)} total rows")
        analyze_data(final_df)