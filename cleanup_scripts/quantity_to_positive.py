import pandas as pd

df = pd.read_csv('filtered_dataset/classification/test/income_fill_file.csv')
df['quantity'] = df['quantity'].abs()
df.to_csv('filtered_dataset/classification/test/quantity_positive_file.csv', index=False)