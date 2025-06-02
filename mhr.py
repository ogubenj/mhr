#pip install streamlit pandas numpy scikit-learn seaborn matplotlib joblib

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import joblib
import warnings
warnings.filterwarnings("ignore")

# Title
st.title("Maternal Health Risk Prediction")

@st.cache_data
def load_data():
    mhr = pd.read_csv(r"C:\Users\User\Downloads\mhr.csv")  # adjust path as needed
    mhr["RiskLevel"] = mhr["RiskLevel"].map({"low risk": 0, "mid risk": 1, "high risk": 2})
    return mhr

# Load Data button
if st.button("Load Data & Overview"):
    st.write("Loading data...")
    mht = load_data()
    st.write(mht.head())
    st.write(mht.describe())
    st.write(mht.isnull().sum())

    st.write("Risk Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="RiskLevel", data=mht, ax=ax)
    ax.set_xticklabels(['Low', 'Mid', 'High'])
    st.pyplot(fig)

    st.write("Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(mht.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.session_state["data"] = mht

# Train models
if "data" in st.session_state:
    if st.button("Train & Evaluate Models"):
        mht = st.session_state["data"]
        X = mht.drop("RiskLevel", axis=1)
        y = mht["RiskLevel"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        def evaluate_model(model, name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)

            st.subheader(name)
            st.write(f"Accuracy: {acc:.4f}")
            st.write(f"F1 Score: {f1:.4f}")
            st.text(classification_report(y_test, y_pred, target_names=["Low", "Mid", "High"]))

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Low", "Mid", "High"],
                        yticklabels=["Low", "Mid", "High"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            return model, acc

        rf = RandomForestClassifier(random_state=42)
        svm = SVC(kernel="rbf")
        log = LogisticRegression()

        models = []
        for mdl, name in [(rf, "Random Forest"), (svm, "Support Vector Machine"), (log, "Logistic Regression")]:
            model, acc = evaluate_model(mdl, name)
            models.append((model, acc, name))

        best_model, best_acc, best_name = max(models, key=lambda x: x[1])
        st.success(f"Best model: {best_name} with accuracy {best_acc:.4f}")

        joblib.dump(best_model, "maternal_health_rf_model.pkl")
        st.session_state["model"] = best_model
        st.session_state["scaler"] = scaler
        st.session_state["X_columns"] = X.columns.tolist()

# Prediction form
if "model" in st.session_state:
    st.header("Predict Maternal Health Risk")

    with st.form(key="predict_form"):
        age = st.number_input("Age", min_value=10, max_value=60, value=30)
        sys_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
        dia_bp = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80)
        sugar = st.number_input("Blood Sugar", min_value=50, max_value=300, value=100)
        temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, format="%.1f")
        hr = st.number_input("Heart Rate", min_value=40, max_value=200, value=70)

        submit = st.form_submit_button("Predict Risk")

    if submit:
        vals = [age, sys_bp, dia_bp, sugar, temp, hr]
        X_input = np.array([vals])
        scaler = st.session_state["scaler"]
        model = st.session_state["model"]

        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]

        risk_map = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
        colors = {0: "green", 1: "orange", 2: "red"}

        def determine_cause(vals, pred):
            age, sys_bp, dia_bp, sugar, temp, hr = vals
            causes = []
            if pred == 0:
                return "No significant risk factors detected."
            if age > 35:
                causes.append("Age > 35 (Consider routine health screening and lifestyle modification)")
            if pred == 2 and sys_bp > 140:
                causes.append("High Systolic BP (>140) (Consult a doctor)")
            elif pred == 1 and sys_bp > 130:
                causes.append("Elevated Systolic BP (>130) (Monitor regularly)")
            if pred == 2 and dia_bp > 90:
                causes.append("High Diastolic BP (>90) (Medical intervention likely needed)")
            elif pred == 1 and dia_bp > 85:
                causes.append("Elevated Diastolic BP (>85) (Lifestyle adjustments recommended)")
            if pred == 2 and sugar > 140:
                causes.append("High Blood Sugar (>140) (Consult a doctor)")
            elif pred == 1 and sugar > 120:
                causes.append("Elevated Blood Sugar (>120) (Reduce sugar intake)")
            if pred == 2 and temp > 38:
                causes.append("High Body Temperature (>38°C) (Seek medical evaluation)")
            elif pred == 1 and temp > 37.5:
                causes.append("Elevated Body Temperature (>37.5°C) (Rest and hydration advised)")
            if pred == 2 and hr > 110:
                causes.append("High Heart Rate (>110 bpm) (Immediate medical assessment)")
            elif pred == 1 and hr > 100:
                causes.append("Elevated Heart Rate (>100 bpm) (Avoid stimulants)")
            if not causes:
                return "Risk factors not clearly identified."
            return ", ".join(causes)

        cause = determine_cause(vals, pred)

        st.markdown(f"<h2 style='color:{colors[pred]};'>🩺 PREDICTED RISK LEVEL: {risk_map[pred].upper()}</h2>", unsafe_allow_html=True)
        st.write("Likely caused by:")
        st.write(cause)

