# Customer Term Deposit Prediction

AI/ML Entry-Level Shortlisting Task — classification model to predict whether a bank
customer will subscribe to a term deposit (`y`), using the UCI **Bank Marketing** dataset
(`bank-full.csv`, 45,211 rows, 17 columns).

Dataset source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Repository contents

```
├── Customer_Term_Deposit_Prediction.ipynb   # Main notebook (EDA → preprocessing → models → evaluation → predictions)
├── data/
│   └── bank-full.csv                        # UCI Bank Marketing dataset (semicolon-separated)
├── outputs/
│   ├── predictions.csv                      # Final predictions: row_id,prediction
│   └── 01–09_*.png                          # Saved EDA / evaluation charts
├── requirements.txt                         # Python dependencies
└── README.md
```

## Approach

1. **EDA** — checked dtypes, `"unknown"` category prevalence (kept as its own category
   rather than imputed), duplicates (none), target imbalance (~88% no / ~12% yes), and
   univariate/bivariate relationships between features and the target.
2. **Preprocessing** — One-Hot Encoding for categoricals, `StandardScaler` for numerics,
   no outlier removal (outliers are real, informative customers, and tree models handle
   them natively), class imbalance handled via `class_weight='balanced'` /
   `scale_pos_weight` rather than resampling, and one engineered feature
   (`was_contacted_before`, derived from `pdays == -1`).
3. **`duration` excluded from the deployed model.** `duration` (last call length) is only
   known *after* a call happens, so using it to decide who to call is data leakage. The
   notebook trains a separate benchmark model *with* `duration` purely to quantify the
   leakage effect — it should not be used to select customers to call.
4. **Model development** — 4 classifiers trained on an identical preprocessing pipeline,
   80/20 stratified train/test split: Logistic Regression, Decision Tree, Random Forest,
   XGBoost.
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices
   for every model.

## Models used & results

(Test set, `duration` excluded — the realistic, deployable setting)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.830 | 0.362 | **0.598** | 0.451 | **0.799** |
| XGBoost | **0.841** | **0.384** | 0.596 | **0.467** | 0.788 |
| Decision Tree | 0.829 | 0.357 | 0.579 | 0.442 | 0.779 |
| Logistic Regression | 0.755 | 0.267 | 0.624 | 0.374 | 0.772 |

(Full run details, plots, and the `duration` leakage comparison are in the notebook.)

## Final model recommendation

**Random Forest.** It gives the best ROC-AUC and a strong recall/F1 balance among the
non-leaking models, is robust to outliers and the mixed numeric/categorical feature set,
and its feature importances (previous campaign success, age, balance, contact recency,
month) are directly explainable to the marketing team.

Top predictive features: `poutcome_success`, `age`, `balance`, `was_contacted_before` /
`pdays`, and campaign `month`.

## `duration` variable

Excluded from the deployed model — it is only known after a call is made, so it cannot be
used to decide *whom to call* without leaking the outcome into the input. See notebook
Section 6 for the quantified comparison with/without it.

## Business scenario answers

1. **Use of predictions:** score the full customer base, rank by predicted probability,
   and prioritize the tele-calling team's limited time toward the highest-probability
   customers.
2. **Optimization target:** not raw accuracy (misleading under ~88/12 imbalance); the
   model is trained to balance precision and recall (F1/ROC-AUC), with a lean toward
   recall since missed subscribers (false negatives) are costlier than one extra wasted
   call (false positive).
3. **False positive vs. false negative impact:** a false positive costs one unnecessary
   call; a false negative is a missed sale — the latter is more costly in this context.
4. **Recommended model:** Random Forest (see above).

## Prediction CSV

`outputs/predictions.csv` follows the required format:

```
row_id,prediction
9,no
16,no
...
```

**Note on the test set:** the assignment brief references "the provided test dataset,"
but no separate test file was included with the brief itself. To keep the pipeline fully
reproducible, `outputs/predictions.csv` is generated from the notebook's own held-out 20%
test split (same rows used for evaluation). `row_id` is the original row index from
`bank-full.csv`, preserved through the split (not used as a model feature).

If an official test file is provided separately, Section 8 of the notebook includes a
ready-to-use snippet — just point it at the new file and re-run; it will use the same
fitted model to generate `outputs/predictions.csv` in the same format.

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure `data/bank-full.csv` is present (already included in this repo).
3. Open and run all cells:
   ```bash
   jupyter notebook Customer_Term_Deposit_Prediction.ipynb
   ```
   or execute end-to-end from the command line:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace Customer_Term_Deposit_Prediction.ipynb
   ```
4. Outputs (`outputs/predictions.csv` and all charts) will be (re)generated in `outputs/`.
