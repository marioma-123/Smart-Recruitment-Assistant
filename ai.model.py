import streamlit as st
import joblib
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Recruitment Assistant", layout="wide")

# =========================
# Load Assets
# =========================
@st.cache_resource
def load_assets():
   base_dir = os.path.dirname(os.path.abspath(__file__))
   model = joblib.load(os.path.join(base_dir, "svm_model.pkl"))
   preprocessor = joblib.load(os.path.join(base_dir, "preprocessor.pkl"))
   return model, preprocessor

model, preprocessor = load_assets()

# =========================
# Shared Logic Function
# =========================
def process_data(input_df):
    df = input_df.copy()
    # Feature Engineering (نفس المنطق الأساسي)
    df["experience_years"] = df["experience"].replace({"<1": 0.5, ">20": 21}).astype(float)
    df["high_experience"] = (df["experience_years"] >= 5).astype(int)
    df["training_per_experience"] = df["training_hours"] / (df["experience_years"] + 1)
    df["large_company"] = df["company_size"].isin(["1000-4999", "5000-9999", "10000+"]).astype(int)
    
    # التنبؤ
    processed_data = preprocessor.transform(df)
    if hasattr(model, "predict_proba"):
     probs = model.predict_proba(processed_data)[:, 1]
    else:
     from scipy.special import expit
    probs = expit(model.decision_function(processed_data))
    df["Probability"] = probs
    df["Predictions"] = (probs >= 0.5).astype(int)
    return df.sort_values(by="Probability", ascending=False)

# =========================
# UI Layout
# =========================
st.title("🤖 Smart Recruitment Assistant")

# Sidebar
st.sidebar.title("Input Options")
option = st.sidebar.radio("Choose Method:", ["Manual Input", "Upload CSV"])

# 1. Manual Input Section
if option == "Manual Input":
    st.subheader("📝 Manual Input")
    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            city = st.text_input("City", "city_103")
            city_index = st.number_input("City Development Index", 0.9)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            exp = st.text_input("Experience", "5")
            company_size = st.selectbox("Company Size", ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
        with col2:
            edu = st.selectbox("Education Level", ["Graduate", "Masters", "Phd"])
            major = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"])
            hours = st.number_input("Training Hours", 0, 100, 40)
            company_type = st.selectbox("Company Type", ["Pvt Ltd", "Funded Startup", "Early Stage Startup", "Public Sector", "NGO", "Other"])
            last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", ">4"])
            
        submitted = st.form_submit_button("Predict Result")
        
        if submitted:
            input_data = pd.DataFrame({
                "city": [city], "city_development_index": [city_index], "gender": [gender],
                "relevent_experience": ["Has relevent experience"], "enrolled_university": ["no_enrollment"],
                "education_level": [edu], "major_discipline": [major], "experience": [exp],
                "company_size": [company_size], "company_type": [company_type],
                "last_new_job": [last_new_job], "training_hours": [hours]
            })
            
            result = process_data(input_data)
            st.metric("Acceptance Probability", f"{result['Probability'].iloc[0]:.2%}")
            st.success(f"Prediction: {'Accepted' if result['Predictions'].iloc[0] == 1 else 'Rejected'}")

# 2. Upload CSV Section
elif option == "Upload CSV":
    st.subheader("☁️ Upload CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        if st.button("⚙️ Process & Dashboard"):
            df_clean = df_raw.drop(["enrollee_id", "target"], axis=1, errors='ignore')
            st.session_state.ranked_df = process_data(df_clean)

    if "ranked_df" in st.session_state:
        df = st.session_state.ranked_df
        
        # Dashboard Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Candidates", len(df))
        c2.metric("Accepted", int(df["Predictions"].sum()))
        c3.metric("Avg Probability", f"{df['Probability'].mean():.2%}")
        
        st.write("---")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            pred_counts = df["Predictions"].replace({1: "Accepted", 0: "Rejected"}).value_counts().reset_index()
            pred_counts.columns = ["Status", "Count"]
            fig_donut = px.pie(pred_counts, names="Status", values="Count", hole=0.5, title="Candidates Acceptance Ratio")
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_chart2:
            heatmap_data = pd.crosstab(df["education_level"], df["company_size"])
            fig_heatmap = px.imshow(heatmap_data, text_auto=True, aspect="auto", title="Heatmap: Education vs Company Size")
            st.plotly_chart(fig_heatmap, use_container_width=True)

        fig_hist = px.histogram(df, x="Probability", nbins=20, title="Probability Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
            
        # --- Expander for Table & Download (مثل الصورة تماماً) ---
        with st.expander(" Candidates Ranking & Raw Data Preview"):
            n = st.slider("Number of rows to display:", 5, len(df), 10)
            st.dataframe(df.head(n))
            
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results CSV", csv, "results.csv", "text/csv")