import pandas as pd
import numpy as np
import os

df = pd.read_csv('filtered_dataset/classification/test/income_fill_file.csv')

df['upsampled'] = False

bins = list(range(0, 1695, 30)) + [np.inf]
df['spend_bin'] = pd.cut(df['spend_amount'], bins=bins)

bin_counts = df['spend_bin'].value_counts()
print("Original bin distribution:")
print(bin_counts.sort_index())

max_count = bin_counts.max()
print(f"\nTarget count per bin: {max_count}")

upsampled_dfs = []

for bin_label in bin_counts.index:
    bin_df = df[df['spend_bin'] == bin_label].copy()
    current_count = len(bin_df)
    if current_count == 0:
        continue
    if current_count < max_count:
    
        n_samples_needed = max_count - current_count
        
        eligible_for_sampling = bin_df[bin_df['inferred'] != True]
        
        if len(eligible_for_sampling) > 0:
            sampled_df = eligible_for_sampling.sample(n=n_samples_needed, replace=True, random_state=919891).copy()
            
            variation = np.random.uniform(-0.05, 0.05, size=len(sampled_df))
            sampled_df['spend_amount'] = sampled_df['spend_amount'] * (1 + variation)
            
            sampled_df['upsampled'] = True
            
            upsampled_dfs.append(sampled_df)
            print(f"\nBin {bin_label}: upsampled {n_samples_needed} entries (from {len(eligible_for_sampling)} eligible rows)")
        else:
            print(f"\nBin {bin_label}: No eligible rows for upsampling (all have inferred=True)")

final_df = pd.concat([df] + upsampled_dfs, ignore_index=True)

final_df = final_df.drop('spend_bin', axis=1)

final_df = final_df.sample(frac=1, random_state=919891).reset_index(drop=True)

print(f"\nOriginal dataset size: {len(df)}")
print(f"Final dataset size: {len(final_df)}")
print(f"Upsampled entries: {final_df['upsampled'].sum()}")

os.makedirs('filtered_dataset/regression/upsampled', exist_ok=True)
final_df.to_csv('filtered_dataset/regression/upsampled/balanced_spend_amount.csv', index=False)
print("\nBalanced dataset saved to: filtered_dataset/regression/upsampled/balanced_spend_amount.csv")