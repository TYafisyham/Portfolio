# Superstore Profit Prediction

Predicting transaction-level profit for a retail superstore to help identify discount policies that erode margins, using regression modeling with a focus on robustness to extreme-loss outliers.
**[Look Jupyter Notebook](./notebooks/superstore.ipynb)**
## Business Problem

Sales teams approve discounts without a clear view of how those discounts affect profitability. Certain sub-categories (e.g., Binders, Machines, Tables) are prone to significant losses when discounted heavily, but this risk is not visible at the point of sale. This project builds a model to estimate the profit of a transaction given its sales amount, discount, quantity, and product/customer attributes — enabling early identification of high-risk discount scenarios.

## Dataset

- **Source:** Sample Superstore dataset (`SampleSuperstore.csv`)
- **Size:** ~9,977 transactions, 13 original columns
- **Target variable:** `Profit`
- **Key features:** `Sales`, `Quantity`, `Discount`, `Category`, `Sub-Category`, `Segment`, `Region`, `Ship Mode`

## Approach

1. **EDA** — analyzed distributions, correlations, and profit performance across category, sub-category, region, and segment. Found that discount is negatively correlated with profit and that a small set of sub-categories drive most losses.
2. **Data splitting** — 60% train / 20% validation / 20% test, split before any preprocessing to avoid leakage.
3. **Feature engineering** — added `Sales_x_Discount` (interaction term) and `Log_Sales` (to reduce skewness).
4. **Preprocessing** — `StandardScaler` for numeric features, `OneHotEncoder` for categorical features, combined via `ColumnTransformer` and fit only on the training set.
5. **Modeling** — compared Linear Regression, Random Forest (default + tuned), Gradient Boosting, XGBoost, and Huber Regressor.
6. **Handling outliers** — investigated extreme-loss transactions (found to be legitimate high-discount deals, not data errors); tested a log-transform of the target, which failed due to exponential error blow-up on inverse transform — documented as a negative result.
7. **Model selection** — selected Huber Regressor for its robustness to outliers and superior MAE on the majority of "normal" transactions.
8. **Evaluation** — predicted-vs-actual and residual plots to validate model behavior on the test set.
9. **Persistence** — final model and preprocessing pipeline saved with `joblib` for reuse without retraining.

## Results (Test Set)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 150.68 | 41.37 | 0.7460 |
| **Huber Regressor (final)** | 159.36 | **29.97** | 0.7159 |
| Random Forest (tuned) | – | – | 0.6274 |
| Gradient Boosting | 173.51 | 28.56 | 0.6641 |
| XGBoost | 205.95 | 27.95 | 0.5269 |

The Huber Regressor was chosen as the final model: it trades a slightly lower R² for a substantially lower MAE, meaning more accurate predictions across the bulk of ordinary transactions rather than being skewed by a handful of extreme-loss cases.

## Key Business Insights

- Higher discounts are consistently associated with lower profit (correlation ≈ -0.22).
- Sub-categories **Tables, Bookcases, and Supplies** are the largest sources of loss.
- **Furniture** is profitable overall but loses money specifically in the **Central** region.
- The `Sales × Discount` interaction is a stronger predictor of profit than either variable alone (highest feature importance across models).

## Project Structure

```
Portfolio/
├── data/
│   ├── raw/                        # original SampleSuperstore.csv
│   └── processed/                  # train/valid/test splits (opsional)
├── notebooks/
│   └── superstore.ipynb            # full EDA + modeling workflow
├── models/                         # saved models (.pkl)
├── reports/
│   └── figures/                    # predicted vs actual, residual plots
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/TYafisyham/Portfolio.git
cd Portfolio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter lab notebooks/superstore.ipynb

## How to Use the Trained Model

```python
import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/huber_model_final.pkl")
preprocessor = joblib.load("models/preprocessor_final.pkl")

new_transaction = pd.DataFrame([{
    "Sales": 500.0,
    "Quantity": 3,
    "Discount": 0.3,
    "Sales_x_Discount": 500.0 * 0.3,
    "Log_Sales": np.log1p(500.0),
    "Category": "Office Supplies",
    "Sub-Category": "Binders",
    "Segment": "Consumer",
    "Region": "South",
    "Ship Mode": "Standard Class"
}])

X = preprocessor.transform(new_transaction)
predicted_profit = model.predict(X)[0]
print(f"Predicted Profit: ${predicted_profit:.2f}")
```

## Limitations

- Model accuracy degrades on transactions with very high discounts (>70%), which are rare in the training data.
- Does not account for temporal factors (seasonality, order date trends) since these were not used as features.
- Dataset size (~6,000 training rows) limits how much tree-based/ensemble models can outperform simpler linear approaches.

## Future Work

- Incorporate order-date-derived features (month, quarter, shipping duration).
- Explore outlier-aware objectives in gradient boosting frameworks (e.g., Huber loss in XGBoost/LightGBM).
- Deploy as a REST API or lightweight dashboard (e.g., Streamlit) for use by the sales team at the point of discount approval.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, matplotlib, Jupyter

## Author

[Yafis] — Data Science Portfolio Project
