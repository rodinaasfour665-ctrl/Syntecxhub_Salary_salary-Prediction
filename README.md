# Salary Prediction | Machine Learning Regression


An end-to-end machine learning workflow to predict salary from professional experience, assessment scores, and relevant categorical features. The project compares a single-feature linear regression baseline with multiple-feature linear models and advanced regression algorithms, tunes selected models, evaluates their performance, and saves the selected pipeline.

> **Dataset transparency:** `salary_data.csv` contains **600 synthetic, educational examples**, not real-world employee salary records. Results are demonstrations of the modeling workflow and should not be presented as real-world predictive performance. `Ecommerce Customers.csv` is a separate, user-provided reference dataset with a different target (`Yearly Amount Spent`); it is not a salary dataset.

## Models compared

- Dummy Mean Baseline
- Single-feature Linear Regression (comparison)
- Multiple-feature Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Random Forest Regressor
- Extra Trees Regressor
- Histogram Gradient Boosting Regressor

Model selection uses cross-validation on the training set. Hyperparameter search covers Ridge, ElasticNet, Random Forest, and Extra Trees.

## Example results

The provided executed notebook, using the **synthetic salary dataset** with `random_state=42`, selected **Ridge Regression** from cross-validation and produced:

| Metric | Result |
| --- | ---: |
| Training cross-validation RMSE | 8,874.11 |
| Test RMSE | 7,996.79 |
| Test MAE | 6,459.79 |
| Test R² | 0.8868 |

These values are specific to the generated demonstration dataset and may change if the data or software versions change. Smaller RMSE/MAE and larger R² generally indicate better prediction performance.

## Repository structure

```text
Syntecxhub_Salary_Prediction/
├── README.md
├── artifacts/
    ├── salary_data.csv
    └── Ecommerce Customers.csv
├── Salary_Prediction_.ipynb
└── artifacts/
    ├── best_salary_regression_model.joblib
    └── salary_model_metadata.joblib
```


## Saved model

The saved `artifacts/best_salary_regression_model.joblib` file contains the **full preprocessing and prediction pipeline**, so it can accept the same raw input feature columns used during training. `artifacts/salary_model_metadata.joblib` stores the selected model name, feature list, target, and evaluation metadata.



## Technologies

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Joblib, Jupyter Notebook.



