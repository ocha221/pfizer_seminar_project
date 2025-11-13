import pandas as pd


df = pd.read_csv('filtered_dataset/classification/test/no_inferred.csv')

for col in ['is_q1', 'is_q2', 'is_q3']:
    if col in df.columns:
        quarter_median = pd.to_datetime(df[df[col] == True]['purchase_date']).median()
        df.loc[(df[col] == True) & (df['purchase_date'].isnull()), 'purchase_date'] = quarter_median

df.to_csv('filtered_dataset/classification/test/date_filled_file.csv', index=False)