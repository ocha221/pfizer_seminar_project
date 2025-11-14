# magnasales
**cleanup_scripts** - preprocessing (cleansing etc) scripts  
**filtered_dataset** - various dataset permutations I tried/used for training (mainly balanced labels, upsampled and some inferred)  
**var_scripts** - random

---

## Models

### Regression

- **regressor-small**: trained on income_fill (9k entries, imputed spend_amount)
- **regressor-large**: trained on balanced_spend_amount (18k entries)

### Classification

- **classifier-small**: trained on balanced_labels (3k entries)
- **classifier-medium**: trained on balanced_is_returning (6k entries, some from imputed spend_amount)
- **classifier-large**: trained on balanced_spend_amount (18k entries, same as regressor-large)

---

`balanced_spend_amount` is produced by:

1. First imputing erroneous spend_amount from the raw combined q1/q2/q3 based on the median (taking into account quantity and magnitude) per product_id
2. The original distribution is pretty uniform, so imputing like that biases it towards the median
3. To remediate this we split the dataset into value bins (steps of 30 seems performant) by spend_amount
4. The bins near the median `bin_median` should have a higher number of entries
5. For each `bin_x` we calculate: `bin_median.entry_count - bin_x.entry_count`
6. Randomly sample `num` ***non-inferred*** entries from within `bin_x` and create synthetic entries till there is equal representation

### Example Entry Structure

```
bin_x_1 = {
    "Name": "Andrew",
    "Spend_amount": "999.8",
    "Date": ...,
    ...
    "is_inferred": "false"
}

bin_x_1_synthetic = {
    "Name": "Andrew",
    "Spend_amount": "999.8" ± rand(-tolerance, +tolerance)  // I used 0.05
    "Date": ...,
    ...
    "upsampled": "true"
}
```
