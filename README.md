# Customer Churn Prediction

An end-to-end customer churn analysis project using the Telco customer dataset. The notebook explores and prepares the data, trains and evaluates classification models, and saves an XGBoost model. A Streamlit app uses the saved model to estimate churn risk for a customer profile.

> Model predictions are estimates based on the provided dataset; they are not guarantees of future customer behavior.

## Project features

- Explore customer demographics, services, contracts, payment methods, and churn.
- Prepare features and train classification models.
- Evaluate predictions, including ROC-AUC, and inspect XGBoost feature importances.
- Save and load models with Joblib.
- Try an interactive churn-risk prediction in the Streamlit app.
- Explore SHAP explanations and a Gradio demo from the notebook.

Models explored in the notebook include Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine, Gaussian Naive Bayes, and XGBoost.

## Project structure

```text
customer-churn-prediction/
├── app.py                       # Streamlit churn prediction app
├── main.ipynb                   # Data analysis, training, and evaluation
├── telco.csv                    # Telco customer dataset
├── best_model.pkl               # Saved model used by the app
├── feature_columns.pkl          # Feature names expected by the saved model
├── joblib_utils.py              # Model save/load and prediction helpers
├── model_utils.py               # Backward-compatible model helper imports
├── churn_feature_importance.png # Feature-importance visualization
├── requirements.txt             # Python dependencies
└── README.md
```

The app expects `best_model.pkl` and `feature_columns.pkl` to be in the project root.

## Requirements

- Python 3.9 or later
- pip

## Setup

Open a terminal in the project directory and create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the prediction app

From the project directory, start Streamlit:

```bash
streamlit run app.py
```

The app opens in your browser. Choose the customer profile in the sidebar and select **Analyze Customer** to view the estimated churn probability, risk category, and key model features.

If the app reports that model files are missing, place `best_model.pkl` and `feature_columns.pkl` in the project directory before starting it.

## Run the notebook

Open `main.ipynb` in Jupyter or VS Code with the Jupyter extension. Run the cells from top to bottom; the notebook reads `telco.csv` from the current project directory.

The notebook saves model files through `joblib_utils.py`. Keep the notebook, dataset, and helper module together so its imports and relative file paths resolve correctly.

## Dataset

`telco.csv` contains customer-level information about demographics, subscribed services, contracts, payment methods, monthly charges, and churn. The notebook uses `Churn Label` as the prediction target and removes identifier and post-churn information from the model features to reduce leakage.

## Model evaluation

The notebook reports classification metrics and ROC-AUC on a held-out test set and displays the most important XGBoost features. Results depend on the dataset and training configuration; review the notebook outputs when retraining the model.
