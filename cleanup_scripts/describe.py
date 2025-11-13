import pandas as pd

df = pd.read_csv('filtered_dataset/classification/test/income_fill_file.csv')

print(df['spend_amount'].describe())