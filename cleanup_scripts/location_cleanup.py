import pandas as pd
import os
df = pd.read_csv('/filtered_dataset/clustering/no_erroneous_income_filtered_spend.csv')

df_cleaned = df.dropna(subset=['lat', 'lon'])

os.makedirs('filtered_dataset/var/age', exist_ok=True)
df_cleaned.to_csv('filtered_dataset/var/age/cleaned_file.csv', index=False)

print(f"Original rows: {len(df)}")
print(f"Cleaned rows: {len(df_cleaned)}")
print(f"Rows dropped: {len(df) - len(df_cleaned)}")