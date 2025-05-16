import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")
st.title("🏫 School Mental Health Prediction Portal")

MODEL_PATH = 'mental_health_rf_model.joblib'
ENCODER_PATH = 'label_encoder.joblib'

# Load model and encoder
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    clf_model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
else:
    st.error("Model or encoder file not found. Please train and save the model using mental.py.")
    st.stop()

st.sidebar.header("Choose User Type")
user_type = st.sidebar.radio("Select your role:", ["School", "Student"])

if user_type == "Student":
    st.header("🧑‍🎓 Student Assessment")
    st.markdown("""
    1. Please fill out the assessment using the following Google Form:
    [Mental Health Assessment Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdFFoCwPIYU4qKvkUQKs6v5kZ_KIWVuQbpFNPMDleF4Eo918w/viewform)
    2. After submitting, download your individual response as a CSV (if available) or request it from your school/teacher.
    3. Upload your response CSV below to see your results.
    """)
    uploaded_student = st.file_uploader("Upload Your Google Form Response CSV", type=["csv"], key="student_csv")
    if uploaded_student is not None:
        df = pd.read_csv(uploaded_student)
        # If only one row, it's the student's response
        if df.shape[0] > 1:
            st.warning("Multiple responses detected. Please upload only your individual response CSV.")
        else:
            # Map answers to risk scores (same logic as app.py)
            risk_score_map = {
                "Always": 1.0, "Often": 0.75, "Sometimes": 0.5,
                "Rarely": 0.25, "Never": 0.0, "Not Sure": 0.5,
                "Skip": 0.5, "Yes": 1.0, "No": 0.0
            }
            disorder_categories = {
                "Stress": ["I feel overwhelmed by my emotions", "I often feel anxious", "I often feel lonely or tearful"],
                "Depression": ["I have felt hopeless or helpless recently", "I feel like life is not worth living", "I have thoughts of hurting myself"],
                "Eating Disorder": ["I worry excessively about gaining weight", "I feel pressure to look a certain way because of social media or peers", "I restrict food intake to control my weight", "I skip meals intentionally", "I eat even when I'm not hungry due to stress or emotions", "I feel guilty after eating", "I avoid eating in front of others", "Do you think your eating habits affect your emotional or physical well-being?"],
                "Behavioral Issues": ["I get into fights with my classmates or friends", "I skip school or classes without a good reason", "I tend to lie or hide the truth to avoid trouble", "I have trouble following rules or instructions", "I find it difficult to share my feelings with others"]
            }
            for disorder, questions in disorder_categories.items():
                df[disorder + " %"] = df[questions].apply(lambda row: np.mean([risk_score_map.get(str(x).strip(), 0.5) for x in row]), axis=1) * 100
            df["Mental Health Risk %"] = df[[col for col in df.columns if col.endswith(" %")]].mean(axis=1)
            X = df[[col for col in df.columns if col.endswith(" %") and col != "Mental Health Risk %"]]
            # Predict risk class
            y_pred = clf_model.predict(X)
            risk_class = le.inverse_transform(y_pred)[0]
            risk_percent = df["Mental Health Risk %"].iloc[0]
            st.success(f"Your Predicted Mental Health Risk %: {risk_percent:.2f}%")
            st.info(f"Your Risk Category: {risk_class}")
            # Guidance (reference app.py logic)
            def personalized_guidance(row):
                guidance = []
                if row["Stress %"] > 60:
                    guidance.append("⚠️ High stress detected. Try relaxation techniques like deep breathing, yoga, or regular physical activity.")
                if row["Depression %"] > 60:
                    guidance.append("🧠 Signs of possible depression. You should speak with a mental health professional.")
                if row["Eating Disorder %"] > 60:
                    guidance.append("🍽️ Eating behavior concerns found. Practice balanced eating, avoid crash dieting, and consider seeing a nutritionist.")
                if row["Behavioral Issues %"] > 60:
                    guidance.append("🧒 Behavioral difficulties noticed. Seek mentorship or conflict resolution workshops.")
                if not guidance:
                    return "✅ You're doing well. Keep practicing positive habits!"
                return " ".join(guidance)
            st.markdown(f"**Guidance:** {personalized_guidance(df.iloc[0])}")
            # Show disorder-wise breakdown
            st.markdown("### Your Disorder-wise Scores")
            st.dataframe(df[[cat + " %" for cat in disorder_categories.keys()]])
    else:
        st.info("After filling the form, upload your response CSV to see your mental health risk report.")

elif user_type == "School":
    st.header("🏫 School Dashboard & Prediction")
    st.markdown("""
    1. Download the Google Form responses as a CSV from Google Forms (File > Download > .csv).
    2. Upload the responses CSV below to generate predictions and dashboards.
    """)
    uploaded_file = st.file_uploader("Upload Google Form Responses CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Try to detect class and gender columns
        possible_class_cols = [col for col in df.columns if col.lower() in ["class", "grade"]]
        class_col = possible_class_cols[0] if possible_class_cols else None
        possible_gender_cols = [col for col in df.columns if col.lower() in ["gender"]]
        gender_col = possible_gender_cols[0] if possible_gender_cols else None

        # Map answers to risk scores (same logic as mental.py)
        risk_score_map = {
            "Always": 1.0, "Often": 0.75, "Sometimes": 0.5,
            "Rarely": 0.25, "Never": 0.0, "Not Sure": 0.5,
            "Skip": 0.5, "Yes": 1.0, "No": 0.0
        }
        disorder_categories = {
            "Stress": ["I feel overwhelmed by my emotions", "I often feel anxious", "I often feel lonely or tearful"],
            "Depression": ["I have felt hopeless or helpless recently", "I feel like life is not worth living", "I have thoughts of hurting myself"],
            "Eating Disorder": ["I worry excessively about gaining weight", "I feel pressure to look a certain way because of social media or peers", "I restrict food intake to control my weight", "I skip meals intentionally", "I eat even when I'm not hungry due to stress or emotions", "I feel guilty after eating", "I avoid eating in front of others", "Do you think your eating habits affect your emotional or physical well-being?"],
            "Behavioral Issues": ["I get into fights with my classmates or friends", "I skip school or classes without a good reason", "I tend to lie or hide the truth to avoid trouble", "I have trouble following rules or instructions", "I find it difficult to share my feelings with others"]
        }
        for disorder, questions in disorder_categories.items():
            df[disorder + " %"] = df[questions].applymap(lambda x: risk_score_map.get(str(x).strip(), 0.5)).mean(axis=1) * 100
        df["Mental Health Risk %"] = df[[col for col in df.columns if col.endswith(" %")]].mean(axis=1)
        X = df[[col for col in df.columns if col.endswith(" %") and col != "Mental Health Risk %"]]
        # Predict risk class
        y_pred = clf_model.predict(X)
        df["Predicted Risk Class"] = le.inverse_transform(y_pred)
        # Save results
        out_csv = df.copy()
        st.download_button("Download Results CSV", out_csv.to_csv(index=False).encode('utf-8'), file_name="Predicted_Student_Risk.csv", mime="text/csv")
        st.success("Predictions generated and results ready for download.")
        # Dashboards
        st.markdown("## 📊 School Mental Health Dashboard")
        st.markdown(f"**Overall Average Mental Health Risk %:** {df['Mental Health Risk %'].mean():.2f}")
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Overall risk distribution
        st.markdown("### Risk Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Mental Health Risk %"], bins=10, kde=True, ax=ax1)
        ax1.set_title("Distribution of Mental Health Risk % (All Students)")
        st.pyplot(fig1)
        # Class-wise
        if class_col:
            st.markdown("### Class-wise Average Risk %")
            class_avg = df.groupby(class_col)["Mental Health Risk %"].mean()
            fig2, ax2 = plt.subplots()
            class_avg.plot(kind='bar', ax=ax2)
            ax2.set_ylabel("Average Risk %")
            ax2.set_title("Class-wise Average Mental Health Risk %")
            st.pyplot(fig2)
        # Gender-wise
        if gender_col:
            st.markdown("### Gender-wise Average Risk %")
            gender_avg = df.groupby(gender_col)["Mental Health Risk %"].mean()
            fig3, ax3 = plt.subplots()
            gender_avg.plot(kind='bar', color=['#66b3ff', '#ffb366'], ax=ax3)
            ax3.set_ylabel("Average Risk %")
            ax3.set_title("Gender-wise Average Mental Health Risk %")
            st.pyplot(fig3)
        # Show table
        st.markdown("### Preview of Results")
        st.dataframe(df.head())
    else:
        st.info("Awaiting CSV upload...")
