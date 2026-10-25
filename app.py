
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071225 0%, #0b1730 55%, #101b35 100%);
    color: #f1f5f9;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}
h1, h2, h3, p, label {
    color: #f1f5f9 !important;
}
.hero {
    background: linear-gradient(120deg, #142858, #10203e, #172a50);
    border: 1px solid #263f72;
    padding: 28px;
    border-radius: 22px;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #a9c4f5 !important;
    font-size: 16px;
}
.panel {
    background: rgba(16, 31, 58, 0.92);
    border: 1px solid #263f72;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}
.metric-card {
    background: linear-gradient(135deg, #152849, #101d37);
    border: 1px solid #2b4778;
    border-radius: 16px;
    padding: 20px;
    min-height: 120px;
}
.metric-label {
    color: #9db6e4;
    font-size: 14px;
}
.metric-value {
    color: #ffffff;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}
div.stButton > button {
    background: linear-gradient(90deg, #6455f5, #0789f9);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    border: 1px solid #8db8ff;
    color: white;
}
[data-testid="stSidebar"] {
    background: #0a1428;
    border-right: 1px solid #263f72;
}
div[data-testid="stMetric"] {
    background: #142542;
    border: 1px solid #2b4778;
    padding: 15px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, columns

try:
    model, feature_columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files missing. Please save best_model.pkl and "
        "feature_columns.pkl from your training notebook."
    )
    st.stop()


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📊 Customer Churn Intelligence</h1>
    <p>AI-powered customer retention and churn risk analytics</p>
    <p>Explore customer attributes, estimate churn probability,
    and identify opportunities for proactive retention.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- FEATURE HELPERS ----------------
def find_column(prefix, category):
    """Find a one-hot column without case-sensitive matching."""
    expected = f"{prefix}_{category}".casefold().strip()
    for col in feature_columns:
        if col.casefold().strip() == expected:
            return col
    return None


def categories_from_columns(prefix, fallbacks):
    """Build dropdown choices from real training columns."""
    values = []
    for col in feature_columns:
        start = prefix + "_"
        if col.casefold().startswith(start.casefold()):
            values.append(col[len(start):])

    # Include known baseline options, which drop_first=True may omit.
    for value in fallbacks:
        if not any(value.casefold() == x.casefold() for x in values):
            values.append(value)

    return values


# ---------------- SIDEBAR INPUTS ----------------
with st.sidebar:
    st.markdown("## 👤 Customer Profile")
    st.caption("Enter customer information to run a prediction.")

    contract_options = categories_from_columns(
        "Contract", ["Month-to-Month", "One Year", "Two Year"]
    )
    internet_options = categories_from_columns(
        "Internet Type", ["DSL", "Fiber Optic", "None"]
    )
    payment_options = categories_from_columns(
        "Payment Method",
        ["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"]
    )

    contract = st.selectbox("📄 Contract Type", contract_options)
    internet_type = st.selectbox("🌐 Internet Type", internet_options)
    payment_method = st.selectbox("💳 Payment Method", payment_options)

    st.markdown("---")
    st.markdown("### 📈 Usage & Loyalty")

    tenure = st.slider("Tenure (months)", 1, 72, 12)
    gb_download = st.slider("Avg Monthly GB Download", 0, 100, 25)
    dependents = st.slider("Number of Dependents", 0, 10, 0)
    referrals = st.slider("Number of Referrals", 0, 15, 0)

    analyze = st.button("✨ Analyze Customer")


# ---------------- BUILD MODEL INPUT ----------------
def make_input():
    row = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    categorical_values = {
        "Contract": contract,
        "Internet Type": internet_type,
        "Payment Method": payment_method
    }

    for prefix, value in categorical_values.items():
        col = find_column(prefix, value)
        if col is not None:
            row.loc[0, col] = 1

    numerical_values = {
        "Avg Monthly GB Download": gb_download,
        "Tenure in Months": tenure,
        "Number of Dependents": dependents,
        "Number of Referrals": referrals
    }

    for col, value in numerical_values.items():
        if col in row.columns:
            row.loc[0, col] = value

    return row


# ---------------- RESULTS ----------------
if analyze:
    try:
        input_data = make_input()
        probability = float(model.predict_proba(input_data)[0][1])
        risk = "HIGH RISK" if probability >= 0.5 else "LOW RISK"
        risk_color = "#fb7185" if probability >= 0.5 else "#34d399"

        st.markdown("## 🎯 Prediction Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Churn Probability</div>
                <div class="metric-value">{probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Risk</div>
                <div class="metric-value" style="color:{risk_color}">
                    {risk}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Likely to Stay</div>
                <div class="metric-value">{1-probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        left, right = st.columns([1, 1])

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### 📊 Probability Breakdown")

            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor("#101d37")
            ax.set_facecolor("#101d37")

            bars = ax.barh(
                ["Likely to Stay", "Likely to Churn"],
                [1 - probability, probability],
                color=["#34d399", "#fb7185"],
                height=0.48
            )

            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability", color="white")
            ax.tick_params(colors="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#526582")
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.0%}")
            )

            for bar, value in zip(bars, [1-probability, probability]):
                ax.text(
                    min(value + 0.02, 0.85),
                    bar.get_y() + bar.get_height()/2,
                    f"{value:.1%}",
                    color="white",
                    va="center",
                    fontweight="bold"
                )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### 💡 Recommended Action")

            if probability >= 0.5:
                st.warning(
                    "This customer is predicted to be at higher churn risk. "
                    "Consider reviewing service concerns, contract options, "
                    "and suitable retention offers."
                )
            else:
                st.success(
                    "This customer is predicted to be at lower churn risk. "
                    "Continue engagement and monitor satisfaction."
                )

            st.markdown("#### Customer Summary")
            st.write(f"**Contract:** {contract}")
            st.write(f"**Internet:** {internet_type}")
            st.write(f"**Payment:** {payment_method}")
            st.write(f"**Tenure:** {tenure} months")
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance for the trained XGBoost model
        if hasattr(model, "feature_importances_"):
            st.markdown("## ⭐ Key Model Features")

            importance = pd.Series(
                model.feature_importances_,
                index=feature_columns
            ).sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#101d37")
            ax.set_facecolor("#101d37")
            importance.sort_values().plot(
                kind="barh", ax=ax, color="#4c9aff"
            )
            ax.tick_params(colors="white")
            ax.set_xlabel("Importance", color="white")
            ax.set_ylabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#526582")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the feature names and categorical labels "
            "match those used during model training."
        )

else:
    st.markdown("""
    <div class="panel">
        <h3>🚀 Ready to Analyze</h3>
        <p>Choose customer attributes in the sidebar and click
        <b>Analyze Customer</b> to see a prediction.</p>
        <p style="color:#9db6e4">
        Results will be based on your saved XGBoost model.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Customer Churn Intelligence • Model predictions are estimates, "
    "not guarantees of future customer behavior."
)
