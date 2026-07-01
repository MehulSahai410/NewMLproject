import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

st.set_page_config(page_title="Placement Predictor", page_icon="🎓")
st.title("🎓 Student Placement Probability Predictor")
st.write("Enter the student details below to calculate the placement probability based on your 6-feature model.")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Initialize session state to hold data logs in memory if it doesn't exist yet
if "logged_data" not in st.session_state:
    st.session_state.logged_data = []

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input("College CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1)
    projects = st.number_input("Number of Projects", min_value=0, max_value=10, value=2)

with col2:
    aptitude_score = st.slider("Aptitude Test Score", min_value=0, max_value=100, value=75)
    soft_skills = st.slider("Soft Skills Rating (out of 5)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    extracurricular = st.selectbox("Participated in Extracurriculars?", ["No", "Yes"])

# Convert Yes/No dropdown to 1/0 numerical value
extracurricular_val = 1 if extracurricular == "Yes" else 0

# SINGLE Predict Button handles both calculation and logging
if st.button("Predict Placement Chance", type="primary"):
    
    features = np.array([[
        cgpa, 
        internships, 
        projects, 
        aptitude_score, 
        soft_skills, 
        extracurricular_val
    ]])
    
    raw_prediction = model.predict(features)[0]
    probability = np.clip(raw_prediction, 0.0, 1.0)
    prob_percentage = round(float(probability) * 100, 2)
    
    st.markdown("---")
    st.subheader("Prediction Analysis:")
    st.metric(label="Calculated Probability", value=f"{prob_percentage}%")
    
    if prob_percentage >= 50.0:
        st.success("🎉 **Model Decision: Placed!** This profile meets the required threshold.")
    elif prob_percentage >= 40.0:
        st.warning("Model Decision: This profile has high chances to secure placements if they improve in one or two parameters")
    else:
        st.error("⚠️ **Model Decision: Not Placed.** Profile metrics might need enhancement.")
        
    # --- MEMORY LOGGING ---
    row_data = {
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "AptitudeScore": aptitude_score,
        "SoftSkills": soft_skills,
        "Extracurricular": extracurricular,
        "PredictedProbability": prob_percentage,
    }
    
    # Append the run details to the session list
    st.session_state.logged_data.append(row_data)



# --- SIDEBAR DOWNLOAD BUTTON ---
# This looks at session memory and renders if rows have been generated
if st.session_state.logged_data:
    st.sidebar.markdown("### 📊 Session Data Log")
    log_df = pd.DataFrame(st.session_state.logged_data)
    st.sidebar.dataframe(log_df)
    
    # Convert dataframe to CSV format for download
    csv_bytes = log_df.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        label="📥 Download Data Log as CSV",
        data=csv_bytes,
        file_name="user_inputs.csv",
        mime="text/csv",
    )