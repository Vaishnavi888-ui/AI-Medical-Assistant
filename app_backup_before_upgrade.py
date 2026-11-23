import joblib
import pandas as pd
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="MediAssist AI",
    page_icon="M",
    layout="wide"
)

# Project paths
PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "models" / "diabetes_model.pkl"
DOCUMENT_PATH = PROJECT_FOLDER / "rag_documents" / "diabetes_who.txt"

# Navigation callbacks
def navigate_to(page):
    st.session_state.page_nav = page

def open_rag_from_home():
    st.session_state.medical_question = st.session_state.home_question
    st.session_state.page_nav = "Medical RAG Assistant"

# Styling
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(120deg, #123b63, #247ba0);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 38px;
}
.hero p {
    color: #e7f3fa;
    font-size: 17px;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("MediAssist AI")
st.sidebar.caption("Your medical information companion")

if "page_nav" not in st.session_state:
    st.session_state.page_nav = "Home"

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Diabetes Prediction", "Medical RAG Assistant", "About Project"],
    key="page_nav"
)

st.sidebar.divider()
st.sidebar.info("Educational prototype only. Not for medical diagnosis.")

# HOME PAGE
if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to MediAssist AI</h1>
        <p>Your smart health information and disease prediction platform.</p>
        <p>Explore machine learning predictions and medical knowledge
        from trusted documents.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("What would you like to do today?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Diabetes Prediction")
        st.write("Explore a machine learning prediction using health measurements.")
        st.button(
            "Open Diabetes Prediction",
            key="home_prediction",
            on_click=navigate_to,
            args=("Diabetes Prediction",)
        )

    with col2:
        st.markdown("### Medical AI Assistant")
        st.write("Search medical information from the WHO diabetes document.")
        st.button(
            "Open Medical Assistant",
            key="home_rag",
            on_click=navigate_to,
            args=("Medical RAG Assistant",)
        )

    st.divider()

    st.subheader("Ask a Medical Question")

    with st.form("home_question_form"):
        st.text_input(
            "What would you like to know?",
            placeholder="Example: What are the symptoms of diabetes?",
            key="home_question"
        )
        st.form_submit_button(
            "Search Medical Information",
            on_click=open_rag_from_home,
            use_container_width=True
        )

    st.divider()

    st.subheader("Explore the Platform")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Main Features", "02")
        st.caption("Prediction and medical information")

    with col2:
        st.metric("ML Model", "Available")
        st.caption("Diabetes prediction model")

    with col3:
        st.metric("Knowledge Source", "WHO")
        st.caption("Diabetes fact sheet")

# DIABETES PREDICTION PAGE
elif page == "Diabetes Prediction":

    st.title("Diabetes Prediction")
    st.write("Enter the health measurements below.")

    if not MODEL_PATH.exists():
        st.error("Model file not found. Check the models folder.")
    else:
        model = joblib.load(MODEL_PATH)

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

            submitted = st.form_submit_button("Predict", use_container_width=True)

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
            st.info(
                "Educational demonstration only. This model is not clinically "
                "validated and must not be used for diagnosis or treatment decisions."
            )

# MEDICAL RAG PAGE
elif page == "Medical RAG Assistant":

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    st.title("Medical RAG Assistant")
    st.write("Ask an educational question using the WHO diabetes document.")

    if not DOCUMENT_PATH.exists():
        st.error("WHO medical document not found. Check the rag_documents folder.")
    else:
        medical_text = DOCUMENT_PATH.read_text(encoding="utf-8")

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
            st.error("No readable text was found.")
        else:
            vectorizer = TfidfVectorizer(stop_words="english")
            document_vectors = vectorizer.fit_transform(chunks)

            with st.form("medical_question_form"):
                question = st.text_input(
                    "Enter your medical question:",
                    placeholder="Example: What are the symptoms of diabetes?",
                    key="medical_question"
                )
                search_button = st.form_submit_button(
                    "Search Medical Documents",
                    use_container_width=True
                )

            if search_button:
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    query_vector = vectorizer.transform([question])
                    scores = cosine_similarity(
                        query_vector, document_vectors
                    ).flatten()

                    top_indices = scores.argsort()[::-1][:3]

                    st.subheader("Retrieved Medical Information")

                    if scores[top_indices[0]] == 0:
                        st.warning("No matching information was found.")
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
                            "These are retrieved passages, not a medical diagnosis "
                            "or personalized medical advice."
                        )

# ABOUT PAGE
elif page == "About Project":

    st.title("About MediAssist AI")

    st.write(
        "Project: AI-Powered Disease Prediction and Medical RAG Assistant"
    )

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Machine Learning
    - Scikit-learn
    - Pandas
    - TF-IDF Information Retrieval
    - Retrieval-Augmented Generation concepts
    - Streamlit
    """)

    st.info(
        "The current medical assistant retrieves relevant passages "
        "from a WHO document. It does not yet generate answers using an LLM."
    )

st.divider()
st.caption(
    "MediAssist AI | Educational prototype only. "
    "Not intended for medical diagnosis or treatment decisions."
)
