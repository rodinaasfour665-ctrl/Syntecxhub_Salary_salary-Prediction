# Salary Prediction | Machine Learning Regression

**Syntecxhub Internship Project**

An end-to-end machine learning workflow to predict salary from professional experience, assessment scores, and relevant categorical features. The project compares a single-feature linear regression baseline with multiple-feature linear models and advanced regression algorithms, tunes selected models, evaluates their performance, and saves the selected pipeline.

> **Dataset transparency:** `salary_data.csv` contains **600 synthetic, educational examples**, not real-world employee salary records. Results are demonstrations of the modeling workflow and should not be presented as real-world predictive performance. `Ecommerce Customers.csv` is a separate, user-provided reference dataset with a different target (`Yearly Amount Spent`); it is not a salary dataset.

## Project objectives

- Explore and clean the dataset, including missing values and duplicated rows.
- Prepare numerical and categorical columns with a reusable preprocessing pipeline.
- Split data into training (80%) and testing (20%) sets.
- Compare single-feature and multiple-feature linear regression.
- Compare several regression algorithms using 5-fold cross-validation.
- Tune selected models using `GridSearchCV`.
- Report **RMSE**, **MAE**, and **R²** on a held-out test set.
- Save the best model and its preprocessing pipeline using Joblib.

## Dataset and features

For **Salary Prediction**, the notebook uses `salary_data.csv`:

| Column | Description |
| --- | --- |
| `YearsExperience` | Years of professional experience |
| `TestScore` | Assessment test score |
| `InterviewScore` | Interview assessment score |
| `Education` | Categorical education level |
| `Department` | Categorical department |
| `Salary` | **Prediction target** |

The alternative **Ecommerce** mode loads `Ecommerce Customers.csv` and predicts `Yearly Amount Spent`, not `Salary`.

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
├── requirements.txt
├── Salary_Prediction_Professional.ipynb
├── Salary_Prediction_Professional_Executed.ipynb
├── salary_data.csv
├── Ecommerce Customers.csv
└── artifacts/
    ├── best_salary_regression_model.joblib
    └── salary_model_metadata.joblib
```

## How to run

1. Clone or download this repository and open its folder.
2. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open `Salary_Prediction_Professional.ipynb` using **Jupyter Notebook**, **JupyterLab**, or **VS Code**.
4. Ensure both CSV files remain in the same folder as the notebook.
5. In the setup cell, use `DATA_MODE = "salary"` to predict salaries, or change it to `"ecommerce"` to run the alternative reference dataset.
6. Select **Run All** to execute the notebook. The notebook creates/updates the `artifacts/` folder and saves the selected model.

**Google Colab:** Upload the notebook and the required CSV file(s) into the Colab working directory, then run the notebook cells in order. Download the resulting model artifact if you want to keep it, since Colab storage can be temporary.

## Saved model

The saved `artifacts/best_salary_regression_model.joblib` file contains the **full preprocessing and prediction pipeline**, so it can accept the same raw input feature columns used during training. `artifacts/salary_model_metadata.joblib` stores the selected model name, feature list, target, and evaluation metadata.

Quick loading example:

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/best_salary_regression_model.joblib")
example = pd.DataFrame([{
    "YearsExperience": 4,
    "TestScore": 78,
    "InterviewScore": 80,
    "Education": "Master",
    "Department": "Finance"
}])
prediction = model.predict(example)
print(f"Predicted salary: {prediction[0]:,.2f}")
```

> Only load Joblib artifacts from sources you trust. For stable deployment, pin the training-time Python and scikit-learn versions.

## Technologies

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Joblib, Jupyter Notebook.

## Internship submission

Repository name: **`Syntecxhub_Salary_Prediction`**. Share the public GitHub repository URL through the internship submission form, following the program's instructions. This is a learning project and does not constitute a production-ready salary decision system.

## Author

**[Your Name]** — Syntecxhub Intern  
GitHub: **[Your GitHub profile URL]**
