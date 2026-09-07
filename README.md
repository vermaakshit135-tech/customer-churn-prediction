# Customer Churn Prediction

Machine learning analysis for predicting customer churn from the Telco customer dataset.

## Project Overview

This project explores customer demographics, services, contract details, charges, and churn behavior. It includes data cleaning, exploratory analysis, feature encoding, scaling, model training, and model evaluation.

## Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors
- Support Vector Machine
- Gaussian Naive Bayes

## Repository Contents

```text
customer-churn-prediction/
├── customer_churn_analysis.ipynb  # Main analysis notebook
├── telco.csv                      # Customer churn dataset
└── README.md                      # Project documentation
```

## Getting Started

### Requirements

- Python 3.9 or later
- Jupyter Notebook or VS Code with the Jupyter extension

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/vermaakshit135-tech/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Open `customer_churn_analysis.ipynb` in Jupyter or VS Code.
3. Run the notebook cells from top to bottom.

The notebook loads `telco.csv` using a relative path, so both files should remain in the project folder.

## Dataset

The dataset contains customer-level information such as:

- Demographics and dependents
- Phone and internet services
- Contract and payment details
- Monthly and total charges
- Churn status and churn reason

The target used for prediction is the customer churn label.

## Goal

The goal is to compare multiple classification models and identify patterns that can help a telecom business understand and reduce customer churn.