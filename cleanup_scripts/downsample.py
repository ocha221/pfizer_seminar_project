import pandas as pd
from sklearn.utils import resample


df = pd.read_csv('filtered_dataset/classification/upsampled/balanced_is_returning.csv')

true_df = df[df['is_returning'] == True]
false_df = df[df['is_returning'] == False]
print(f"Original True rows: {len(true_df)}")
print(f"Original False rows: {len(false_df)}")
"""
if len(true_df) > len(false_df):
    true_df = resample(true_df, n_samples=len(false_df), random_state=919891)
else:
    false_df = resample(false_df, n_samples=len(true_df), random_state=919891)

# Combine
balanced_df = pd.concat([true_df, false_df]).sample(frac=1, random_state=919891)
os.makedirs('filtered_dataset/regression/downsampled', exist_ok=True)
balanced_df.to_csv('filtered_dataset/regression/downsampled/balanced_age_file.csv', index=False)"""