# Customer Term Deposit Prediction

Predicts whether a bank customer will subscribe to a term deposit (`y`), using the UCI
**Bank Marketing** dataset (`bank-full.csv`, 45,211 rows, 17 columns). The project covers
the full pipeline — EDA, preprocessing, model training/evaluation in a notebook — plus a
deployed web app (FastAPI/Flask-style backend + frontend) that serves live predictions
from the trained model.

- **Repo:** https://github.com/puneetraghav-dev/Customer-Term-Deposit-Prediction
- **Live demo:** https://customer-term-deposit-prediction-mo.vercel.app/
- **Dataset source:** https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## Repository Structure

```
├── Customer_Term_Deposit_Prediction.ipynb   # Main notebook: EDA → preprocessing → models → evaluation
├── bank-full.csv                            # UCI Bank Marketing dataset (semicolon-separated)
├── backend/                                 # API that loads the trained model and serves predictions
├── frontend/                                # Web UI (deployed to Vercel) that calls the backend
├── LICENSE                                  # MIT License
└── README.md
```

---

## Approach

1. **EDA** — Checked data types, prevalence of the `"unknown"` category (kept as its own
   category rather than imputed), duplicates (none found), target class imbalance
   (~88% "no" / ~12% "yes"), and univariate/bivariate relationships between features and
   the target.
2. **Preprocessing** — One-Hot Encoding for categorical features, `StandardScaler` for
   numeric features, no outlier removal (outliers represent real, informative customers
   and tree-based models handle them natively), class imbalance handled via
   `class_weight='balanced'` / `scale_pos_weight` rather than resampling, and one
   engineered feature — `was_contacted_before` (derived from `pdays == -1`).
3. **`duration` excluded from the deployed model.** `duration` (last call length) is only
   known *after* a call happens, so using it to decide who to call would leak future
   information into the input. A separate benchmark model *with* `duration` was trained
   purely to quantify the leakage effect — it is **not** used for deployment or scoring
   customers to call.
4. **Model development** — Four classifiers trained on an identical preprocessing
   pipeline with an 80/20 stratified train/test split.
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices
   computed for every model.
6. **Deployment** — The best-performing, leakage-free model was serialized and wrapped in
   a backend API; the frontend collects customer attributes and displays the predicted
   subscription likelihood, deployed live on Vercel.

## Models Used

| Model                   | Type                              |
|--------------------------|-----------------------------------|
| Logistic Regression      | Linear baseline                   |
| Decision Tree            | Single tree, class-weighted       |
| Random Forest            | Bagged ensemble of trees          |
| XGBoost                  | Gradient-boosted ensemble         |

## Results

(Test set, `duration` excluded — the realistic, deployable setting)

| Model                | Accuracy  | Precision | Recall    | F1-score  | ROC-AUC   |
|-----------------------|-----------|-----------|-----------|-----------|-----------|
| **Random Forest**     | 0.830     | 0.362     | **0.598** | 0.451     | **0.799** |
| XGBoost               | **0.841** | **0.384** | 0.596     | **0.467** | 0.788     |
| Decision Tree         | 0.829     | 0.357     | 0.579     | 0.442     | 0.779     |
| Logistic Regression   | 0.755     | 0.267     | 0.624     | 0.374     | 0.772     |

Full run details, plots, and the `duration`-leakage comparison are documented in the notebook.

## Final Model Recommendation

**Random Forest** is the model used in production.

- Best ROC-AUC and a strong recall/F1 balance among the non-leaking models.
- Robust to outliers and to the mixed numeric/categorical feature set without heavy tuning.
- Feature importances are directly explainable to a marketing team: `poutcome_success`,
  `age`, `balance`, `was_contacted_before` / `pdays`, and campaign `month` are the top
  predictors.
- Business framing: since missed subscribers (false negatives) are costlier than an extra
  wasted call (false positives), the model favors recall over raw accuracy, and Random
  Forest gives the best trade-off among leakage-free models.

## Instructions to Run the Project

### 1. Run the notebook (EDA, training, evaluation)

```bash
git clone https://github.com/puneetraghav-dev/Customer-Term-Deposit-Prediction.git
cd Customer-Term-Deposit-Prediction

pip install -r requirements.txt   # if not present, install: pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

jupyter notebook Customer_Term_Deposit_Prediction.ipynb
```

Or run it end-to-end from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace Customer_Term_Deposit_Prediction.ipynb
```

Make sure `bank-full.csv` is present in the repo root (it's already included).

### 2. Run the backend locally

```bash
cd backend
pip install -r requirements.txt     # Python backend
# or
npm install                         # Node/Express backend

# start the API (adjust to the actual entry point in backend/, e.g.):
python app.py
# or
uvicorn main:app --reload
# or
npm start
```

By default the API should come up on `http://localhost:8000` (or whatever port is
configured in `backend/`). Check `backend/` for the exact start script and required
environment variables (e.g. model path, `PORT`, CORS origin).

### 3. Run the frontend locally

```bash
cd frontend
npm install
npm run dev
```

Point the frontend's API base URL (usually an `.env` / `.env.local` variable such as
`VITE_API_URL` or `NEXT_PUBLIC_API_URL`) at your local backend URL from step 2, then open
the printed local URL (typically `http://localhost:3000` or `http://localhost:5173`) in
your browser.

### 4. Use the live app

No setup needed — try it directly at:
👉 **https://customer-term-deposit-prediction-mo.vercel.app/**

---

## Business Scenario Notes

1. **Use of predictions:** Score the full customer base, rank by predicted probability,
   and prioritize the tele-calling team's limited time toward the highest-probability
   customers.
2. **Optimization target:** Not raw accuracy (misleading under ~88/12 class imbalance).
   The model balances precision and recall (F1 / ROC-AUC), leaning toward recall since a
   missed subscriber is costlier than one extra wasted call.
3. **False positive vs. false negative impact:** A false positive costs one unnecessary
   call; a false negative is a missed sale — the latter is more costly in this context.
4. **Recommended model:** Random Forest (see above).

## License

Released under the [MIT License](./LICENSE).
