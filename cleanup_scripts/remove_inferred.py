import pandas as pd
import os

df = pd.read_csv('filtered_dataset/classification/test/income_fill_file.csv')
if not 'inferred' in df.columns:
    print("No inferred column")
    exit(0)

df = df[df['inferred'] == False]
os.makedirs('filtered_dataset/classification/test', exist_ok=True)
df.to_csv('filtered_dataset/classification/test/no_inferred.csv', index=False)
