import joblib
import pandas as pd


MODEL_FILENAME = "best_model.pkl"


def save_model(model, filename):
    joblib.dump(model, filename)
    print("Model saved successfully!")


def load_model(filename=MODEL_FILENAME):
    return joblib.load(filename)


def get_feature_importances(model, feature_names):
    return pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)


def predict_churn(customer_data, model=None):
    if model is None:
        model = load_model()

    probability = model.predict_proba(customer_data)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return prediction, probability
