import joblib
import pandas as pd
from pathlib import Path

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# Main title
st.title("🩺 AI-Powered Medical Assistant")

st.subheader("Disease Prediction & Medical Information Assistant")

st.write(
    "Welcome! This educational application demonstrates "
    "machine learning-based diabetes prediction and "
    "AI-powered medical information assistance."
)

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "Home",
        "Diabetes Prediction",
        "Medical RAG Assistant",
        "About Project"
    ]
)

# Home page
if page == "Home":

    st.header("Welcome to AI Medical Assistant")

    st.write(
        "This project combines Machine Learning, "
        "Generative AI, and Retrieval-Augmented Generation (RAG)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info("ML  Machine Learning Disease Prediction")

    with col2:
        st.info("RAG Medical Knowledge Assistant")

# Diabetes prediction page
elif page == "Diabetes Prediction":

    st.header("🩺 Diabetes Prediction")
    st.write("Enter sample health measurements below.")

    project_folder = Path(__file__).resolve().parent
    model_path = project_folder / "models" / "diabetes_model.pkl"
    model = joblib.load(model_path)

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 2)
            glucose = st.number_input("Glucose", 1.0, 300.0, 120.0)
            blood_pressure = st.number_input("Blood Pressure", 1.0, 200.0, 70.0)
            skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, 25.0)

        with col2:
            insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
            bmi = st.number_input("BMI", 1.0, 80.0, 28.5)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 21, 100, 30)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning("Model prediction: Positive class (Outcome 1)")
        else:
            st.success("Model prediction: Negative class (Outcome 0)")

        st.write(f"Positive-class model probability: {probability * 100:.2f}%")

        st.info("Educational demonstration only. This model is not clinically validated and must not be used for diagnosis or treatment decisions.")
# Medical RAG page
elif page == "Medical RAG Assistant":

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    st.header("Medical RAG Assistant")
    st.write("Ask an educational medical question using the WHO diabetes document.")

    project_folder = Path(__file__).resolve().parent
    document_path = project_folder / "rag_documents" / "diabetes_who.txt"

    if not document_path.exists():
        st.error("WHO medical document not found. Please check the rag_documents folder.")
    else:
        medical_text = document_path.read_text(encoding="utf-8")

        chunk_size = 500
        overlap = 100
        chunks = []

        start = 0
        while start < len(medical_text):
            chunk = medical_text[start:start + chunk_size].strip()

            if chunk:
                chunks.append(chunk)

            start += chunk_size - overlap

        if not chunks:
            st.error("No readable text was found in the medical document.")
        else:
            vectorizer = TfidfVectorizer(stop_words="english")
            document_vectors = vectorizer.fit_transform(chunks)

            with st.form("medical_question_form"):
                question = st.text_input(
                    "Enter your medical question:",
                    placeholder="Example: What are the symptoms of diabetes?"
                )

                search_button = st.form_submit_button("Search Medical Documents")

            if search_button:
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    query_vector = vectorizer.transform([question])
                    scores = cosine_similarity(
                        query_vector,
                        document_vectors
                    ).flatten()

                    top_indices = scores.argsort()[::-1][:3]

                    st.subheader("Retrieved Medical Information")

                    if scores[top_indices[0]] == 0:
                        st.warning("No matching information was found in the document.")
                    else:
                        for rank, index in enumerate(top_indices, start=1):
                            if scores[index] <= 0:
                                continue

                            with st.expander(
                                f"Result {rank} - Similarity: {scores[index]:.3f}",
                                expanded=True
                            ):
                                st.write(chunks[index])

                        st.caption("Source: WHO Diabetes Fact Sheet")
                        st.info(
                            "These are retrieved passages from the source document, "
                            "not a medical diagnosis or personalized medical advice."
                        )
# About page
elif page == "About Project":

    st.header("About This Project")

    st.write("""
    Project: AI-Powered Disease Prediction & Medical RAG Assistant

    Technologies:
    - Python
    - Machine Learning
    - Scikit-learn
    - Generative AI
    - Retrieval-Augmented Generation
    - Streamlit
    """)

st.divider()

st.caption(
    "Educational prototype only. Not intended for medical diagnosis "
    "or treatment decisions."
)





