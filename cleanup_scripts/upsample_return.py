import pandas as pd
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import LabelEncoder
import os
DATE_FORMAT = 'mixed'
df = pd.read_csv('filtered_dataset/classification/test/date_filled_file.csv')


if 'upsampled' in df.columns:
    df = df.drop('upsampled', axis=1)
if 'inferred' in df.columns:
    df = df[df['inferred'] == False]
    df = df.drop('inferred', axis=1)

df['upsampled'] = False


categorical_cols = ['gender', 'payment_method', 'is_q1', 'is_q2', 'is_q3']
numerical_cols = ['age', 'income', 'quantity', 'spend_amount', 'lat', 'lon']

X = df.drop(['is_returning', 'upsampled'], axis=1)
y = df['is_returning']
print("Checking for missing values:")
print(X.isnull().sum())

X['purchase_date_ordinal'] = (pd.to_datetime(X['purchase_date'], format=DATE_FORMAT) - pd.Timestamp('1970-01-01')).dt.days
X_encoded = X.copy()

label_encoders = {}
for col in categorical_cols:
    if col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

if 'inferred' in X_encoded.columns:
    X_encoded['inferred'] = X_encoded['inferred'].astype(int)

X_encoded = X_encoded.drop('purchase_date', axis=1)
categorical_features = [i for i, col in enumerate(X_encoded.columns) 
                       if col in categorical_cols + ['inferred', 'product_id', 'customer_id', 'is_q1', 'is_q2', 'is_q3']]

print(f"Original class distribution:")
print(y.value_counts())

smoten = SMOTEN(random_state=919891, k_neighbors=5)
X_resampled, y_resampled = smoten.fit_resample(X_encoded, y)

print(f"\nResampled class distribution:")
print(pd.Series(y_resampled).value_counts())
df_resampled = pd.DataFrame(X_resampled, columns=X_encoded.columns)

for col, le in label_encoders.items():
    if col in df_resampled.columns:
        df_resampled[col] = le.inverse_transform(df_resampled[col].astype(int))

df_resampled['purchase_date'] = pd.Timestamp('1970-01-01') + pd.to_timedelta(df_resampled['purchase_date_ordinal'], unit='D')
df_resampled = df_resampled.drop('purchase_date_ordinal', axis=1)
df_resampled['is_returning'] = y_resampled


df_resampled['upsampled'] = False
df_resampled.iloc[len(df):, df_resampled.columns.get_loc('upsampled')] = True

if 'inferred' in df_resampled.columns:
    df_resampled['inferred'] = df_resampled['inferred'].astype(bool)

df_resampled = df_resampled.sample(frac=1, random_state=919891).reset_index(drop=True)

print(f"\nOriginal dataset size: {len(df)}")
print(f"Final dataset size: {len(df_resampled)}")
print(f"Upsampled entries: {df_resampled['upsampled'].sum()}")

os.makedirs('filtered_dataset/classification/upsampled', exist_ok=True)
df_resampled.to_csv('filtered_dataset/classification/upsampled/balanced_is_returning.csv', index=False)
print("\nBalanced dataset saved to: filtered_dataset/classification/upsampled/balanced_is_returning.csv")