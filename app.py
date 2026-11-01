
import os
import json
import pickle
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    ROOT, "models", "v2", "xgboost.pkl"
)

ASSET_PATH = os.path.join(
    ROOT, "results", "v2", "dashboard_data.json"
)

METRICS_PATH = os.path.join(
    ROOT, "results", "v2", "final_test_metrics.json"
)

RESULTS_DIR = os.path.join(
    ROOT, "results", "v2"
)

DEFAULT_THRESHOLD = 0.70

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111a2e;
    border-right: 1px solid #25324a;
}

[data-testid="stSidebar"] h1 {
    color: #6ee7b7;
}

h1, h2, h3 {
    color: #f8fafc;
}

[data-testid="stMetric"] {
    background: #141f33;
    border: 1px solid #293750;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #9ca3af;
}

[data-testid="stMetricValue"] {
    color: #6ee7b7;
}

.stButton > button {
    background: #0f9f78;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background: #0b8061;
    color: white;
    border: none;
}

.stDownloadButton > button {
    background: #1d2b43;
    color: #d1fae5;
    border: 1px solid #31536a;
    border-radius: 10px;
}

.stDownloadButton > button:hover {
    border-color: #6ee7b7;
    color: white;
}

div[data-testid="stFileUploader"] {
    background: #141f33;
    border: 1px dashed #3b536b;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stExpander"] {
    background: #141f33;
    border-radius: 12px;
    border: 1px solid #293750;
}

hr {
    border-color: #293750;
}

.small-muted {
    color: #94a3b8;
    font-size: 0.9rem;
}

.hero {
    background: linear-gradient(120deg, #123c3a, #14253b);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #245b55;
    margin-bottom: 22px;
}

.hero h1 {
    color: #d1fae5;
    margin-bottom: 5px;
}

.hero p {
    color: #b7d8d0;
    margin-bottom: 0;
}

.section-label {
    color: #6ee7b7;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


@st.cache_data
def load_image(path):
    with open(path, "rb") as file:
        return file.read()


model = load_model()
dashboard = load_json(ASSET_PATH)
test_metrics = load_json(METRICS_PATH)

FEATURES = dashboard["feature_names"]


# =========================================================
# COMMON HELPERS
# =========================================================

def section_title(title, subtitle=None):
    st.markdown(f"## {title}")

    if subtitle:
        st.markdown(
            f'<p class="small-muted">{subtitle}</p>',
            unsafe_allow_html=True
        )


def make_chart(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d1d5db"),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        gridcolor="#293750",
        zerolinecolor="#293750"
    )

    fig.update_yaxes(
        gridcolor="#293750",
        zerolinecolor="#293750"
    )

    return fig


def format_probability(value):
    return f"{value * 100:.2f}%"


def create_sample_csv():
    sample = pd.DataFrame(
        [[0.0] * len(FEATURES)],
        columns=FEATURES
    )

    sample["Time"] = 50000
    sample["Amount"] = 100

    return sample.to_csv(index=False).encode("utf-8")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🛡️ FraudGuard AI")
    st.caption("Intelligent Transaction Risk Analysis")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Fraud Detection",
            "Analytics & Insights",
            "Documentation"
        ]
    )

    st.divider()

    st.markdown("### Model status")
    st.success("XGBoost model loaded")

    st.markdown("### Model configuration")
    st.write("**Model:** XGBoost")
    st.write("**Default threshold:** 0.70")

    st.divider()

    st.caption("FraudGuard AI V2 | Portfolio Project")


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <div class="section-label">TRANSACTION INTELLIGENCE PLATFORM</div>
        <h1>FraudGuard AI</h1>
        <p>
            Detect suspicious financial transactions using
            machine learning and ensemble-based classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_title(
        "Dataset overview",
        "A summary of the cleaned credit card transaction dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total transactions",
        f"{dashboard['total_transactions']:,}"
    )

    c2.metric(
        "Genuine transactions",
        f"{dashboard['genuine_transactions']:,}"
    )

    c3.metric(
        "Fraudulent transactions",
        f"{dashboard['fraud_transactions']:,}"
    )

    c4.metric(
        "Fraud percentage",
        f"{dashboard['fraud_percentage']:.3f}%"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        section_title("Transaction distribution")

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Genuine", "Fraud"],
                    values=[
                        dashboard["genuine_transactions"],
                        dashboard["fraud_transactions"]
                    ],
                    hole=0.65,
                    marker=dict(
                        colors=["#14b8a6", "#f97373"]
                    ),
                    textinfo="label+percent"
                )
            ]
        )

        fig.update_layout(
            showlegend=False,
            annotations=[
                dict(
                    text="Transactions",
                    x=0.5,
                    y=0.5,
                    font_size=15,
                    showarrow=False,
                    font_color="#e5e7eb"
                )
            ]
        )

        st.plotly_chart(
            make_chart(fig),
            use_container_width=True
        )

    with col2:

        section_title("Transaction amount profile")

        st.metric(
            "Average amount",
            f"{dashboard['mean_amount']:.2f}"
        )

        st.metric(
            "Median amount",
            f"{dashboard['median_amount']:.2f}"
        )

        st.metric(
            "Maximum amount",
            f"{dashboard['max_amount']:.2f}"
        )

        st.caption(
            "Amounts are shown in the dataset's original units."
        )

    st.divider()

    section_title("Final model performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Precision",
        f"{test_metrics['Precision'] * 100:.2f}%"
    )

    c2.metric(
        "Recall",
        f"{test_metrics['Recall'] * 100:.2f}%"
    )

    c3.metric(
        "F1 score",
        f"{test_metrics['F1 Score'] * 100:.2f}%"
    )

    c4.metric(
        "PR-AUC",
        f"{test_metrics['PR-AUC'] * 100:.2f}%"
    )

    st.caption(
        "Metrics are from the untouched 15% test set, "
        "using XGBoost at threshold 0.70."
    )

    st.info(
        "This dashboard uses an anonymized credit card fraud "
        "dataset. It is a machine learning demonstration, "
        "not a production banking fraud prevention system."
    )


# =========================================================
# PAGE 2: FRAUD DETECTION
# =========================================================

elif page == "Fraud Detection":

    section_title(
        "Fraud detection",
        "Upload transaction data to calculate fraud probabilities."
    )

    st.markdown("""
    Upload a CSV containing the 30 model input features.
    If your file includes a `Class` column, it will be ignored.
    """)

    with st.expander("How does the threshold work?"):

        st.write("""
        The threshold controls when a transaction is classified
        as fraudulent.

        For example, a threshold of 0.70 means that transactions
        with a predicted fraud probability of 70% or higher
        are classified as fraud.

        Lower thresholds generally identify more fraud but may
        create more false alarms. Higher thresholds generally
        reduce false alarms but may miss more fraudulent cases.
        """)

    threshold = st.slider(
        "Fraud classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        format="%.2f"
    )

    st.caption(
        f"Current classification threshold: {threshold:.2f}"
    )

    st.download_button(
        "Download sample CSV template",
        data=create_sample_csv(),
        file_name="fraudguard_sample_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            # Ignore target column if present
            if "Class" in uploaded_df.columns:
                uploaded_df = uploaded_df.drop(
                    columns=["Class"]
                )

            missing_columns = [
                col for col in FEATURES
                if col not in uploaded_df.columns
            ]

            extra_columns = [
                col for col in uploaded_df.columns
                if col not in FEATURES
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            elif uploaded_df.empty:

                st.warning("The uploaded CSV contains no transactions.")

            else:

                # Keep only required columns in model order
                input_df = uploaded_df[FEATURES].copy()

                # Validate numeric values
                for col in FEATURES:
                    input_df[col] = pd.to_numeric(
                        input_df[col],
                        errors="coerce"
                    )

                if input_df.isnull().any().any():

                    st.error(
                        "The uploaded file contains missing "
                        "or non-numeric feature values. "
                        "Please clean the file and upload again."
                    )

                elif not np.isfinite(
                    input_df.to_numpy(dtype=float)
                ).all():

                    st.error(
                        "The uploaded file contains infinite values."
                    )

                else:

                    if extra_columns:
                        st.warning(
                            "Extra columns were ignored: "
                            + ", ".join(extra_columns)
                        )

                    probabilities = model.predict_proba(
                        input_df
                    )[:, 1]

                    predictions = (
                        probabilities >= threshold
                    ).astype(int)

                    results = uploaded_df.copy()

                    results["Fraud Probability"] = probabilities
                    results["Prediction"] = np.where(
                        predictions == 1,
                        "Fraud",
                        "Genuine"
                    )

                    results["Risk Level"] = np.select(
                        [
                            probabilities < 0.30,
                            probabilities < 0.70
                        ],
                        [
                            "Low",
                            "Medium"
                        ],
                        default="High"
                    )

                    fraud_count = int(
                        (predictions == 1).sum()
                    )

                    genuine_count = int(
                        (predictions == 0).sum()
                    )

                    st.divider()

                    section_title("Analysis results")

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Transactions analyzed",
                        f"{len(results):,}"
                    )

                    c2.metric(
                        "Fraud alerts",
                        f"{fraud_count:,}"
                    )

                    c3.metric(
                        "Genuine transactions",
                        f"{genuine_count:,}"
                    )

                    st.divider()

                    col1, col2 = st.columns(2)

                    with col1:

                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=[
                                        "Genuine",
                                        "Fraud"
                                    ],
                                    values=[
                                        genuine_count,
                                        fraud_count
                                    ],
                                    hole=0.65,
                                    marker=dict(
                                        colors=[
                                            "#14b8a6",
                                            "#f97373"
                                        ]
                                    )
                                )
                            ]
                        )

                        st.plotly_chart(
                            make_chart(fig),
                            use_container_width=True
                        )

                    with col2:

                        fig = px.histogram(
                            results,
                            x="Fraud Probability",
                            nbins=25,
                            title="Fraud probability distribution",
                            color_discrete_sequence=["#38bdf8"]
                        )

                        st.plotly_chart(
                            make_chart(fig),
                            use_container_width=True
                        )

                    st.subheader("Prediction results")

                    st.dataframe(
                        results.style.format({
                            "Fraud Probability": "{:.2%}"
                        }),
                        use_container_width=True,
                        height=400
                    )

                    st.subheader("Highest-risk transactions")

                    high_risk = results.sort_values(
                        "Fraud Probability",
                        ascending=False
                    ).head(10)

                    st.dataframe(
                        high_risk,
                        use_container_width=True
                    )

                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download prediction results",
                        data=csv_data,
                        file_name="fraudguard_predictions.csv",
                        mime="text/csv"
                    )

                    st.success(
                        "Transaction analysis completed successfully."
                    )

                    st.caption(
                        "Predictions are model estimates, not verified "
                        "fraud findings. Uploaded labels, if present, "
                        "are not used for prediction or evaluation."
                    )

        except Exception as error:

            st.error(
                f"Unable to process the uploaded CSV: {error}"
            )


# =========================================================
# PAGE 3: ANALYTICS & INSIGHTS
# =========================================================

elif page == "Analytics & Insights":

    section_title(
        "Analytics & insights",
        "Explore transaction patterns and model evaluation results."
    )

    amount_data = dashboard["amount_distribution"]

    amount_df = pd.DataFrame({
        "Amount (approx.)": amount_data["labels"],
        "Genuine": amount_data["genuine"],
        "Fraud": amount_data["fraud"]
    })

    col1, col2 = st.columns(2)

    with col1:

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=amount_df["Amount (approx.)"],
                y=amount_df["Genuine"],
                name="Genuine",
                marker_color="#14b8a6"
            )
        )

        fig.add_trace(
            go.Bar(
                x=amount_df["Amount (approx.)"],
                y=amount_df["Fraud"],
                name="Fraud",
                marker_color="#f97373"
            )
        )

        fig.update_layout(
            title="Transaction amount distribution",
            barmode="overlay",
            xaxis_title="Approximate amount (log-scaled bins)",
            yaxis_title="Number of transactions"
        )

        st.plotly_chart(
            make_chart(fig),
            use_container_width=True
        )

    with col2:

        time_data = dashboard["time_distribution"]

        time_df = pd.DataFrame({
            "Elapsed time (hours)": time_data["labels"],
            "Genuine": time_data["genuine"],
            "Fraud": time_data["fraud"]
        })

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time_df["Elapsed time (hours)"],
                y=time_df["Genuine"],
                name="Genuine",
                mode="lines",
                line=dict(color="#14b8a6", width=2)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time_df["Elapsed time (hours)"],
                y=time_df["Fraud"],
                name="Fraud",
                mode="lines",
                line=dict(color="#f97373", width=2)
            )
        )

        fig.update_layout(
            title="Transactions over elapsed time",
            xaxis_title="Elapsed hours since first recorded transaction",
            yaxis_title="Number of transactions"
        )

        st.plotly_chart(
            make_chart(fig),
            use_container_width=True
        )

    st.divider()

    section_title(
        "Feature importance",
        "Top features used by the selected XGBoost model."
    )

    importance = dashboard["feature_importance"]

    importance_df = pd.DataFrame({
        "Feature": importance["features"],
        "Importance": importance["values"]
    }).sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 features by XGBoost gain",
        color="Importance",
        color_continuous_scale="Teal"
    )

    st.plotly_chart(
        make_chart(fig, height=550),
        use_container_width=True
    )

    st.caption(
        "Feature importance indicates model usage, not causation. "
        "V1–V28 are anonymized PCA-derived features."
    )

    st.divider()

    section_title(
        "Final test-set evaluation",
        "Performance on 42,559 previously untouched test transactions."
    )

    metric_columns = st.columns(5)

    metric_items = [
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1 Score", "F1 Score"),
        ("ROC-AUC", "ROC-AUC"),
        ("PR-AUC", "PR-AUC")
    ]

    for column, (label, key) in zip(
        metric_columns,
        metric_items
    ):

        column.metric(
            label,
            f"{test_metrics[key] * 100:.2f}%"
        )

    st.caption(
        "The classification threshold used for these test results was 0.70."
    )

    col1, col2 = st.columns(2)

    confusion_path = os.path.join(
        RESULTS_DIR,
        "final_test_confusion_matrix.png"
    )

    roc_path = os.path.join(
        RESULTS_DIR,
        "final_test_roc_curve.png"
    )

    pr_path = os.path.join(
        RESULTS_DIR,
        "final_test_pr_curve.png"
    )

    with col1:

        st.subheader("Confusion matrix")

        if os.path.exists(confusion_path):
            st.image(
                load_image(confusion_path),
                use_container_width=True
            )

    with col2:

        st.subheader("ROC curve")

        if os.path.exists(roc_path):
            st.image(
                load_image(roc_path),
                use_container_width=True
            )

    st.subheader("Precision-Recall curve")

    if os.path.exists(pr_path):
        st.image(
            load_image(pr_path),
            use_container_width=True
        )

    with st.expander("View exact final test metrics"):

        st.json(test_metrics)


# =========================================================
# PAGE 4: DOCUMENTATION
# =========================================================

elif page == "Documentation":

    section_title(
        "Documentation",
        "A guide to FraudGuard AI and its features."
    )

    st.subheader("1. What is FraudGuard AI?")

    st.write("""
    FraudGuard AI is a machine learning application designed
    to identify potentially fraudulent credit card transactions.

    It uses an XGBoost classification model trained on an
    anonymized credit card transaction dataset.
    """)

    st.subheader("2. Dataset")

    st.write("""
    The project uses the Credit Card Fraud Detection dataset
    from Kaggle, originally containing 284,807 transactions.

    After removing 1,081 exact duplicate copies, the cleaned
    dataset contains 283,726 transactions, including 473
    fraudulent transactions.

    The dataset contains 30 input features and one target column.
    """)

    st.subheader("3. Machine learning model")

    st.write("""
    XGBoost is a gradient-boosted decision tree algorithm.
    It builds multiple decision trees sequentially, with
    later trees improving on errors made by earlier trees.

    Class imbalance was handled using the scale_pos_weight
    parameter during training.
    """)

    st.subheader("4. Why is the dataset imbalanced?")

    st.write("""
    Fraudulent transactions make up only a very small fraction
    of the dataset. A model that predicts every transaction
    as genuine could achieve high accuracy while failing to
    detect fraud.

    Therefore, this project focuses on precision, recall,
    F1 score and Precision-Recall AUC rather than accuracy alone.
    """)

    st.subheader("5. What is the threshold?")

    st.write("""
    The threshold converts the model's predicted probability
    into a binary classification.

    For example, with a threshold of 0.70:

    - Probability below 0.70: Genuine prediction.
    - Probability equal to or above 0.70: Fraud prediction.

    Adjusting the threshold changes the balance between
    detecting fraud and generating false alarms.
    """)

    st.subheader("6. Understanding the metrics")

    metric_descriptions = {
        "Precision": "Of all transactions flagged as fraud, how many were actually fraudulent?",
        "Recall": "Of all actual fraud cases, how many did the model detect?",
        "F1 Score": "A combined measure of precision and recall.",
        "ROC-AUC": "Measures how well the model separates the two classes across thresholds.",
        "PR-AUC": "Summarizes the precision-recall trade-off, especially useful for imbalanced datasets."
    }

    for name, description in metric_descriptions.items():

        with st.expander(name):
            st.write(description)

    st.subheader("7. How to use Fraud Detection")

    st.write("""
    1. Open the Fraud Detection page.
    2. Download the sample CSV template if needed.
    3. Prepare a CSV containing all 30 required input features.
    4. Upload the CSV.
    5. Adjust the threshold if required.
    6. Review fraud probabilities and classifications.
    7. Download the prediction results.
    """)

    st.subheader("8. Important limitations")

    st.warning("""
    - The dataset is anonymized and does not contain merchant,
      customer or location information.
    - V1–V28 are PCA-derived features and cannot be interpreted
      as ordinary business attributes.
    - The model is a portfolio demonstration, not a production
      financial security system.
    - Predictions are probabilities, not confirmed fraud findings.
    - The test evaluation uses a stratified random split;
      temporal generalization has not been established.
    - Model performance can change on data from other sources
      or different time periods.
    """)

    st.subheader("9. Technology stack")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - XGBoost
    - Imbalanced-learn
    - Streamlit
    - Plotly
    - Matplotlib and Seaborn
    """)

    st.subheader("10. Project information")

    st.write("""
    FraudGuard AI V2 is an educational portfolio project
    demonstrating data preprocessing, imbalanced learning,
    ensemble model comparison, threshold selection,
    explainability and interactive deployment.
    """)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:10px;">
        FraudGuard AI V2 · Machine Learning Portfolio Project
        <br>
        Predictions are estimates and should not be treated
        as verified financial fraud decisions.
    </div>
    """,
    unsafe_allow_html=True
)
