import streamlit as st
from joblib import load
import pandas as pd
import pandasql as ps
import plotly.graph_objects as go
from ml import get_raw_dataset, get_encoded_dataset, model_path, scaler_path


st.set_page_config(
    page_title="Diabetes Prediction Studio", page_icon="🩺", layout="wide"
)


@st.cache_data
def get_cached_data():
    raw_dataset = get_raw_dataset()
    encoded_dataset = get_encoded_dataset()
    knn_reg_loaded = load(model_path)
    scaler_loaded = load(scaler_path)

    return raw_dataset, encoded_dataset, knn_reg_loaded, scaler_loaded


raw_dataset, encoded_dataset, knn_reg_loaded, scaler_loaded = get_cached_data()


feature_columns = ["preg", "plas", "pres",
                   "skin", "insu", "mass", "pedi", "age"]
outcome_labels = {0: "Tested Negative", 1: "Tested Positive"}


def run_query(sql):
    return ps.sqldf(sql, {"raw": raw_dataset.copy(), "encoded": encoded_dataset.copy()})


def predict_diabetes(feature_values):
    feature_frame = pd.DataFrame([feature_values], columns=feature_columns)
    feature_scaled = scaler_loaded.transform(feature_frame)
    probabilities = knn_reg_loaded.predict_proba(feature_scaled)[0]
    prediction = int(probabilities.argmax())
    return {
        "prediction": prediction,
        "risk_probability": float(probabilities[1]),
        "confidence": float(probabilities[prediction]),
        "features": feature_frame,
    }


st.title("🩺 Diabetes Prediction")
st.markdown(
    "A data-driven assistant that combines machine learning, exploratory analytics, and interactive visuals to support diabetes screening."
)

predict_tab, explore_tab = st.tabs(["Patient Assessment", "Explore Dataset"])

with predict_tab:
    st.header("Patient Assessment")
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.selectbox(
                "Number of Pregnancies",
                options=list(range(0, 21)),
                index=0,
            )

            glucose = st.slider(
                "Glucose Level (mg/dL)",
                min_value=50,
                max_value=250,
                value=117,
                step=1,
            )

            blood_pressure = st.slider(
                "Blood Pressure (mm Hg)",
                min_value=40,
                max_value=140,
                value=72,
                step=1,
            )

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                value=23.0,
                help="Measured triceps skinfold thickness",
            )

        with col2:
            insulin = st.number_input(
                "Insulin (µU/mL)",
                min_value=0.0,
                max_value=900.0,
                step=1.0,
                value=30.5,
                help="2-hour serum insulin level",
            )

            weight = st.slider(
                "Weight (kg)",
                min_value=30.0,
                max_value=200.0,
                value=70.0,
                step=0.5,
            )

            height = st.number_input(
                "Height (cm)",
                min_value=100.0,
                max_value=220.0,
                step=0.1,
                value=170.0,
            )

            dpf = st.slider(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=2.5,
                step=0.01,
                value=0.37,
                help="Family history influence on diabetes risk",
            )

            age = st.selectbox(
                "Age",
                options=list(range(10, 81)),
                index=19,
            )

        bmi = round(weight / ((height / 100) ** 2), 1)
        st.caption(f"Calculated BMI (mass): **{bmi}**")

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        feature_values = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
        ]
        prediction_results = predict_diabetes(feature_values)
        confidence_pct = prediction_results["confidence"] * 100
        predicted_label = f"{outcome_labels[prediction_results['prediction']]}"
        color_positive = "#c0392b"
        color_negative = "#16a085"
        if prediction_results["prediction"] == 0:
            st.success("Analysis complete.")
        else:
            st.error("Elevated risk identified.")
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            st.metric("Outcome", predicted_label, border=True)
        with c2:
            st.metric("Model Confidence",
                      f"{confidence_pct:.1f} %", border=True)
        with c3:
            if prediction_results["prediction"] == 0:
                recomendation = "Maintaining a healthy lifestyle."
            else:
                recomendation = "Consult a healthcare professional."

            st.metric("Medical Recomendation", recomendation, border=True)

        st.subheader("Input Snapshot")
        st.dataframe(
            prediction_results["features"].rename(
                columns={
                    "preg": "Pregnancies",
                    "plas": "Glucose",
                    "pres": "Blood Pressure",
                    "skin": "Skin Thickness",
                    "insu": "Insulin",
                    "mass": "BMI (mass)",
                    "pedi": "Diabetes Pedigree Function",
                    "age": "Age",
                }
            )
        )

with explore_tab:
    st.header("Dataset Intelligence")
    total_records = len(raw_dataset)
    positive_cases = int((raw_dataset["class"] == "tested_positive").sum())
    avg_bmi = float(raw_dataset["mass"].mean())
    avg_glucose = float(raw_dataset["plas"].mean())
    metric_cols = st.columns(4)
    metric_cols[0].metric("Records", f"{total_records}")
    metric_cols[1].metric("Positive Cases", f"{positive_cases}")
    metric_cols[2].metric("Average BMI", f"{avg_bmi:.1f}")
    metric_cols[3].metric("Average Glucose", f"{avg_glucose:.1f} mg/dL")
    sql_overview = run_query(
        """
        SELECT class AS outcome,
               COUNT(*) AS patients,
               AVG(plas) AS avg_glucose,
               AVG(mass) AS avg_bmi,
               AVG(age) AS avg_age
        FROM raw
        GROUP BY class
        ORDER BY patients DESC
        """
    )
    st.subheader("Outcomes by Group (SQL Powered)")
    st.dataframe(
        sql_overview.style.format(
            {"avg_glucose": "{:.1f}", "avg_bmi": "{:.1f}", "avg_age": "{:.1f}"}
        )
    )
    class_distribution = raw_dataset["class"].value_counts().reset_index()
    class_distribution.columns = ["Outcome", "Count"]
    bar_chart = go.Figure(
        data=[
            go.Bar(
                x=class_distribution["Outcome"],
                y=class_distribution["Count"],
                marker=dict(color=["#16a085", "#c0392b"]),
            )
        ]
    )
    bar_chart.update_layout(
        title="Outcome Distribution",
        xaxis_title="Outcome",
        yaxis_title="Number of Patients",
        template="plotly_white",
    )
    st.plotly_chart(bar_chart, use_container_width=True)
    st.subheader("Feature Spotlight")
    feature_summary = (
        raw_dataset[["plas", "pres", "skin", "insu", "mass", "pedi", "age"]]
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .rename(
            index={
                "mean": "Mean",
                "std": "Std Dev",
                "min": "Minimum",
                "max": "Maximum",
            },
            columns={
                "plas": "Glucose",
                "pres": "Blood Pressure",
                "skin": "Skin Thickness",
                "insu": "Insulin",
                "mass": "BMI",
                "pedi": "DPF",
                "age": "Age",
            },
        )
    )
    st.dataframe(feature_summary)
    st.subheader("Raw Records")
    st.dataframe(raw_dataset)
