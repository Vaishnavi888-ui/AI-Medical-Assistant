
import joblib
import pandas as pd
import streamlit as st
from google import genai
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediAssist AI",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "diabetes_model.pkl"
DATA_PATH = BASE / "dataset" / "diabetes.csv"
DOC_PATH = BASE / "rag_documents" / "diabetes_who.txt"


# ============================================================
# ACCESSIBLE COLOR THEME
# ============================================================

st.markdown("""
<style>

/* -----------------------------------------------------------
   1. MAIN BACKGROUND AND DEFAULT TEXT

   Background: #F4F7FB (very light blue-grey)
   Text:       #263548 (dark navy-grey)

   Dark text on a light background improves readability.
----------------------------------------------------------- */

.stApp {
    background-color: #F4F7FB !important;
    color: #263548 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #F4F7FB !important;
}

[data-testid="stMain"] {
    background-color: #F4F7FB !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* -----------------------------------------------------------
   2. HEADINGS AND NORMAL CONTENT

   Headings: #163653 (dark blue)
   Body:     #263548 (dark grey-blue)

   These colors are darker than the original pale text.
----------------------------------------------------------- */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5 {
    color: #163653 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] label {
    color: #263548 !important;
}

.stApp label,
.stApp [data-testid="stCaptionContainer"] {
    color: #526477 !important;
}


/* -----------------------------------------------------------
   3. HERO BANNER

   Background: dark blue-green gradient
   Text:       white (#FFFFFF)

   White text contrasts strongly with the dark banner.
----------------------------------------------------------- */

.hero {
    background: linear-gradient(120deg, #12365A, #087E8B);
    padding: 38px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(20, 50, 80, 0.12);
}

/* Force all hero text to remain white, even if global rules change. */
.hero,
.hero h1,
.hero h2,
.hero p,
.hero span,
.hero strong {
    color: #FFFFFF !important;
}

.hero h1 {
    font-size: 38px;
}


/* -----------------------------------------------------------
   4. DASHBOARD CARDS AND CONTAINERS

   Card background: #FFFFFF
   Card text:       #263548
   Border:          #D6E0EA

   White cards with dark text remain readable on the
   light blue-grey page background.
----------------------------------------------------------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6E0EA !important;
    border-radius: 15px !important;
    padding: 12px !important;
}

/* Explicitly keep text inside cards dark. */
[data-testid="stVerticalBlockBorderWrapper"]
[data-testid="stMarkdownContainer"] p,
[data-testid="stVerticalBlockBorderWrapper"]
[data-testid="stMarkdownContainer"] h1,
[data-testid="stVerticalBlockBorderWrapper"]
[data-testid="stMarkdownContainer"] h2,
[data-testid="stVerticalBlockBorderWrapper"]
[data-testid="stMarkdownContainer"] h3 {
    color: #263548 !important;
}


/* -----------------------------------------------------------
   5. SIDEBAR

   Sidebar background: #E8EFF7
   Sidebar text:       #24364B

   Dark text on the pale sidebar makes navigation readable.
----------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background-color: #E8EFF7 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #24364B !important;
}

/* Sidebar radio option text. */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #24364B !important;
}


/* -----------------------------------------------------------
   6. BUTTONS

   Button background: #126B83
   Button text:       #FFFFFF

   White text on dark teal creates a clear button label.
----------------------------------------------------------- */

.stButton > button,
.stFormSubmitButton > button {
    background-color: #126B83 !important;
    color: #FFFFFF !important;
    border: 1px solid #126B83 !important;
    border-radius: 10px !important;
    min-height: 46px;
    font-weight: 600;
}

/* Keep nested button text white as well. */
.stButton > button p,
.stButton > button span,
.stFormSubmitButton > button p,
.stFormSubmitButton > button span {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #0D5268 !important;
    border-color: #0D5268 !important;
    color: #FFFFFF !important;
}


/* -----------------------------------------------------------
   7. METRIC CARDS

   Background: #FFFFFF
   Labels:     #526477
   Values:     #163653

   Dark metric values make important numbers stand out.
----------------------------------------------------------- */

[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6E0EA !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #526477 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #163653 !important;
}


/* -----------------------------------------------------------
   8. INPUTS AND TEXT AREAS

   Input background: #FFFFFF
   Input text:       #263548
   Input border:     #AABACA

   Dark input text prevents typed content from disappearing.
----------------------------------------------------------- */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stDateInput input {
    background-color: #FFFFFF !important;
    color: #263548 !important;
    border: 1px solid #AABACA !important;
    border-radius: 8px !important;
    caret-color: #126B83 !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #66788A !important;
    opacity: 1 !important;
}

/* Dropdown controls. */
.stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #263548 !important;
}


/* -----------------------------------------------------------
   9. EXPANDERS

   Expander background: #FFFFFF
   Expander text:       #263548
   Border:              #D6E0EA
----------------------------------------------------------- */

[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6E0EA !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #163653 !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #263548 !important;
}


/* -----------------------------------------------------------
   10. ALERTS AND INFORMATION BOXES

   Keep dark text on pale alert backgrounds.
   The alert background is left to Streamlit so its
   warning, success and information meanings are preserved.
----------------------------------------------------------- */

/* Accessible alert styling: pale background and dark readable text */
[data-testid="stAlert"] {
    background-color: #E8F1FA !important;
    border: 1px solid #9DB9D3 !important;
    border-radius: 12px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] li,
[data-testid="stAlert"] strong {
    color: #163653 !important;
}


/* -----------------------------------------------------------
   11. DATAFRAMES AND TABLES

   Table background: white
   Table text:       dark navy-grey
   Header background: #E8EFF7
----------------------------------------------------------- */

[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
}

[data-testid="stDataFrame"] * {
    color: #263548 !important;
}


/* -----------------------------------------------------------
   12. CHAT INTERFACE

   Chat content uses dark text for comfortable reading.
----------------------------------------------------------- */

[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6E0EA !important;
    border-radius: 12px !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span {
    color: #263548 !important;
}

[data-testid="stChatInput"] textarea {
    color: #263548 !important;
    background-color: #FFFFFF !important;
}


/* -----------------------------------------------------------
   13. FOOTER AND DIVIDERS

   Muted grey-blue footer text remains readable.
----------------------------------------------------------- */

hr {
    border-color: #D6E0EA !important;
}

/* FIX SIDEBAR SUCCESS & INFO TEXT */
section[data-testid="stSidebar"] [data-testid="stAlert"],
section[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}
/* ALL SUCCESS, WARNING, INFO, ERROR BOX TEXT */
[data-testid="stAlert"] * {
    color: #FFFFFF !important;
}
/* ALL SUCCESS, WARNING, INFO, ERROR BOX TEXT */
[data-testid="stAlert"] * {
    color: #FFFFFF !important;
}

/* ALERTS */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* Force ALL alert text to white */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div,
[data-testid="stAlert"] li,
[data-testid="stAlert"] strong,
[data-testid="stAlert"] *,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #24364B !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


@st.cache_data
def load_dataset(path, modified):
    return pd.read_csv(path)


@st.cache_data
def load_chunks(path, modified):
    text = Path(path).read_text(encoding="utf-8")

    chunks = []
    size = 500
    overlap = 100
    start = 0

    while start < len(text):
        part = text[start:start + size].strip()

        if part:
            chunks.append(part)

        start += size - overlap

    return chunks


def get_retriever(chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(chunks)

    return vectorizer, vectors


def generate_gemini_answer(question, sources, chat_history=None):
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from Streamlit secrets.")

    client = genai.Client(api_key=api_key)

    context = "\n\n".join(sources)

    recent_history = (chat_history or [])[-6:]
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
    )

    prompt = f"""
You are an educational medical information assistant.

Answer the user's question using only the supplied WHO document excerpts.
If the excerpts do not contain enough information, clearly say so.
Do not diagnose diseases or prescribe treatment.
Encourage users to consult a qualified healthcare professional for personal medical concerns.

Recent conversation:
{history_text}

User's latest question:
{question}

WHO document excerpts:
{context}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text or "I could not generate an answer. Please try again."

def go_to(page):
    st.session_state["navigation"] = page


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("MediAssist AI")
st.sidebar.caption("Intelligent Health Information Platform")
st.sidebar.divider()

pages = [
    "Home",
    "Diabetes Prediction",
    "Medical AI Assistant",
    "Dataset Analytics",
    "BMI Calculator",
    "Health Education",
    "About Project"
]

if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Home"

if "navigation_target" in st.session_state:
    st.session_state["navigation"] = st.session_state.pop("navigation_target")

page = st.sidebar.radio(
    "Explore the application",
    pages,
    key="navigation"
)

st.sidebar.divider()

st.sidebar.success("Application ready")

st.sidebar.info(
    "For educational use only. "
    "This application cannot diagnose disease."
)


# ============================================================
# 1. HOME DASHBOARD
# ============================================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to MediAssist AI</h1>
        <p>Your intelligent platform for disease prediction,
        medical information retrieval and health education.</p>
        <p>Explore the tools below to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Your Health Information Workspace")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Core Tools", "06")

    with c2:
        status = "Available" if MODEL_PATH.exists() else "Missing"
        st.metric("Prediction Model", status)

    with c3:
        status = "Available" if DOC_PATH.exists() else "Missing"
        st.metric("WHO Document", status)

    st.divider()

    st.subheader("What Would You Like to Explore?")

    cards = [
        (
            "Diabetes Prediction",
            "Explore a machine learning prediction using health measurements.",
            "Open Prediction"
        ),
        (
            "Medical AI Assistant",
            "Search the WHO diabetes document with your questions.",
            "Ask a Question"
        ),
        (
            "Dataset Analytics",
            "Explore dataset distributions and feature patterns.",
            "View Analytics"
        ),
        (
            "BMI Calculator",
            "Calculate BMI using height and weight.",
            "Calculate BMI"
        ),
        (
            "Health Education",
            "Learn about diabetes, symptoms and prevention.",
            "Explore Education"
        ),
        (
            "About Project",
            "Understand the project and technologies used.",
            "Project Details"
        )
    ]

    for i in range(0, len(cards), 2):

        left, right = st.columns(2)

        for col, item in zip((left, right), cards[i:i + 2]):

            name, description, button = item

            with col:

                with st.container(border=True):

                    st.subheader(name)
                    st.write(description)

                    st.button(
                        button,
                        key="home_" + name,
                        on_click=go_to,
                        args=(name,),
                        use_container_width=True
                    )

    st.divider()

    st.subheader("Ask a Medical Question")

    with st.form("home_search"):

        home_question = st.text_input(
            "Enter a question about diabetes",
            placeholder="Example: What are common symptoms of diabetes?"
        )

        submitted = st.form_submit_button(
            "Search Medical Information",
            use_container_width=True
        )

    if submitted:

        if home_question.strip():

            st.session_state["pending_question"] = home_question.strip()
            st.session_state["navigation_target"] = "Medical AI Assistant"
            st.rerun()

        else:
            st.warning("Please enter a question first.")

    st.caption(
        "MediAssist AI is an academic prototype, "
        "not a medical diagnostic service."
    )


# ============================================================
# 2. DIABETES PREDICTION
# ============================================================

elif page == "Diabetes Prediction":

    st.title("Diabetes Prediction")

    st.caption(
        "Enter measurements to see the output of the trained "
        "machine learning model."
    )

    if not MODEL_PATH.exists():

        st.error("Model file not found: models/diabetes_model.pkl")

    else:

        model = load_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime
        )

        st.info(
            "This model is an educational prototype "
            "and is not clinically validated."
        )

        with st.form("prediction_form"):

            st.subheader("Patient Measurements")

            col1, col2 = st.columns(2)

            with col1:

                pregnancies = st.number_input(
                    "Pregnancies", 0, 20, 2
                )

                glucose = st.number_input(
                    "Glucose", 1.0, 300.0, 120.0
                )

                bp = st.number_input(
                    "Blood Pressure", 1.0, 200.0, 70.0
                )

                skin = st.number_input(
                    "Skin Thickness", 0.0, 100.0, 25.0
                )

            with col2:

                insulin = st.number_input(
                    "Insulin", 0.0, 1000.0, 80.0
                )

                bmi = st.number_input(
                    "BMI", 1.0, 80.0, 28.5
                )

                pedigree = st.number_input(
                    "Diabetes Pedigree Function",
                    0.0, 3.0, 0.5
                )

                age = st.number_input(
                    "Age", 21, 100, 30
                )

            submitted = st.form_submit_button(
                "Run Prediction",
                use_container_width=True
            )

        if submitted:

            values = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age
            }])

            prediction = model.predict(values)[0]

            probability = model.predict_proba(values)[0][1]

            st.divider()
            st.subheader("Model Output")

            result_col, probability_col = st.columns(2)

            with result_col:

                if prediction == 1:

                    st.warning(
                        "Predicted class: Positive (Outcome 1)"
                    )

                else:

                    st.success(
                        "Predicted class: Negative (Outcome 0)"
                    )

            with probability_col:

                st.metric(
                    "Positive-Class Model Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(float(probability))

            st.caption(
                "This is a model probability, not a calibrated "
                "personal risk estimate."
            )

            st.warning(
                "This model is not clinically validated. "
                "It must not be used for diagnosis or treatment decisions."
            )


# ============================================================
            # Explainable AI: Feature Contributions
            st.divider()
            st.subheader("Why did the model make this prediction?")

            transformed_values = model[:-1].transform(values)
            classifier = model.named_steps["classifier"]
            contributions = transformed_values[0] * classifier.coef_[0]

            explanation = pd.DataFrame({
                "Feature": values.columns,
                "Contribution": contributions
            })

            explanation["Direction"] = explanation["Contribution"].apply(
                lambda x: "Increases model score" if x > 0
                else "Decreases model score"
            )

            explanation["Absolute Impact"] = explanation["Contribution"].abs()
            explanation = explanation.sort_values("Absolute Impact", ascending=False)

            st.caption("These are mathematical model contributions, not medical causes.")
            st.write("### Feature Contribution Chart")
            st.bar_chart(explanation.set_index("Feature")["Contribution"])

            st.write("### Simple Prediction Explanation")
            st.write(
                "Here are the three measurements that contributed most "
                "to the model's prediction:"
            )

            top_features = explanation.head(3)

            for _, item in top_features.iterrows():
                direction = (
                    "pushed the model score upward"
                    if item["Contribution"] > 0
                    else "pushed the model score downward"
                )

                feature_name = item["Feature"].replace(
                    "DiabetesPedigreeFunction",
                    "Family history indicator"
                )

                st.markdown(
                    f"- **{feature_name}:** This measurement {direction}."
                )

            st.info(
                "These explanations describe how the AI model used the "
                "measurements. They do not identify medical causes or "
                "confirm whether someone has diabetes."
            )
            # Professional PDF Prediction Report
            import io
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            )

            pdf_buffer = io.BytesIO()
            pdf = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=45,
                leftMargin=45,
                topMargin=45,
                bottomMargin=45
            )

            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph(
                "AI Medical Assistant",
                styles["Title"]
            ))
            story.append(Paragraph(
                "Diabetes Prediction Report",
                styles["Heading2"]
            ))
            story.append(Spacer(1, 18))

            result_text = "Positive" if prediction == 1 else "Negative"

            story.append(Paragraph("Prediction Summary", styles["Heading2"]))

            summary_data = [
                ["Prediction Result", result_text],
                ["Positive-Class Model Probability", f"{probability * 100:.2f}%"]
            ]

            summary_table = Table(summary_data, colWidths=[230, 220])
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 9),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 18))

            story.append(Paragraph("Patient Measurements", styles["Heading2"]))

            measurement_data = [["Measurement", "Entered Value"]]
            for column in values.columns:
                measurement_data.append([
                    str(column),
                    str(values.iloc[0][column])
                ])

            measurement_table = Table(
                measurement_data,
                colWidths=[230, 220]
            )
            measurement_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]))
            story.append(measurement_table)
            story.append(Spacer(1, 18))

            story.append(Paragraph("Prediction Explanation", styles["Heading2"]))

            for _, item in explanation.head(3).iterrows():
                direction = (
                    "pushed the model score upward"
                    if item["Contribution"] > 0
                    else "pushed the model score downward"
                )
                story.append(Paragraph(
                    f"{item['Feature']}: {direction}.",
                    styles["BodyText"]
                ))
                story.append(Spacer(1, 5))

            story.append(Spacer(1, 15))
            story.append(Paragraph("Medical Disclaimer", styles["Heading2"]))
            story.append(Paragraph(
                "This report is generated by an educational AI prototype. "
                "The model is not clinically validated and must not be used "
                "for diagnosis or treatment decisions. The model probability "
                "is not a calibrated personal risk estimate. Please consult "
                "a qualified healthcare professional for medical advice.",
                styles["BodyText"]
            ))

            pdf.build(story)
            pdf_buffer.seek(0)

            st.download_button(
                label="Download PDF Prediction Report",
                data=pdf_buffer.getvalue(),
                file_name="diabetes_prediction_report.pdf",
                mime="application/pdf"
            )
            st.subheader("Risk Assessment")

            if prediction == 1:
                st.error("⚠️ Higher Diabetes Likelihood Indicated")

                st.markdown(f"""
                Based on the information provided, the model predicts
                **Outcome 1 (positive classification)**.

                **Positive-class model probability:** {probability * 100:.1f}%

                ### Recommendations

                - Consult a healthcare professional for appropriate testing.
                - Discuss blood glucose monitoring with your healthcare professional.
                - Maintain healthy eating and exercise habits.
                - Do not rely on this result alone for diagnosis.
                """)

            else:
                st.success("✅ Lower Diabetes Likelihood Indicated")

                st.markdown(f"""
                Based on the information provided, the model predicts
                **Outcome 0 (negative classification)**.

                **Positive-class model probability:** {probability * 100:.1f}%

                ### Recommendations

                - Continue maintaining healthy eating habits.
                - Exercise regularly.
                - Attend routine health checkups.
                - A negative prediction does not rule out diabetes.
                """)

# 3. MEDICAL AI ASSISTANT
# ============================================================

elif page == "Medical AI Assistant":

    st.title("Medical AI Assistant")

    st.caption(
        "Ask questions and retrieve relevant passages "
        "from the WHO diabetes fact sheet."
    )

    if not DOC_PATH.exists():

        st.error(
            "WHO document not found in rag_documents/diabetes_who.txt"
        )

    else:

        chunks = load_chunks(
            str(DOC_PATH),
            DOC_PATH.stat().st_mtime
        )

        if not chunks:

            st.error("The medical document contains no readable text.")

        else:

            vectorizer, vectors = get_retriever(chunks)

            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            for message in st.session_state["chat_history"]:

                with st.chat_message(message["role"]):

                    st.markdown(message["content"])

                    if message.get("sources"):

                        with st.expander("Retrieved Source Passages"):

                            source_scores = message.get("source_scores", [])

                            for number, passage in enumerate(
                                message["sources"], 1
                            ):
                                st.markdown(f"**WHO Source Passage {number}**")

                                if number <= len(source_scores):
                                    st.caption(
                                        f"Retrieval relevance score: {source_scores[number - 1]:.3f}"
                                    )

                                st.write(passage)

            initial_question = st.session_state.pop(
                "pending_question", None
            )

            question = st.chat_input(
                "Ask a question about diabetes..."
            )

            if initial_question:
                question = initial_question

            if question:

                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": question
                })

                previous_questions = [
                    message["content"]
                    for message in st.session_state["chat_history"][:-1]
                    if message["role"] == "user"
                ][-2:]

                retrieval_query = " ".join(previous_questions + [question])
                query_vector = vectorizer.transform([retrieval_query])

                scores = cosine_similarity(
                    query_vector,
                    vectors
                ).flatten()

                indices = scores.argsort()[::-1][:3]

                if scores[indices[0]] <= 0:

                    answer = (
                        "I could not find a relevant passage in the "
                        "WHO document. Please try rephrasing your question."
                    )

                    sources = []

                else:

                    selected_indices = [
                        i for i in indices if scores[i] > 0
                    ]

                    sources = [chunks[i] for i in selected_indices]
                    source_scores = [
                        float(scores[i]) for i in selected_indices
                    ]

                    try:
                        with st.spinner("Gemini is preparing your answer..."):
                            answer = generate_gemini_answer(
                                question,
                                sources,
                                st.session_state["chat_history"][:-1]
                            )

                    except Exception as e:
                        answer = (
                            "Gemini could not generate an answer. "
                            "Please check the error details below."
                        )
                        st.error("Gemini API Error")
                        st.exception(e)
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "source_scores": source_scores if sources else []
                })

                st.rerun()

            if st.button("Clear Conversation"):

                st.session_state["chat_history"] = []
                st.rerun()

            st.caption("Knowledge source: WHO Diabetes Fact Sheet. Answers are generated by Gemini using retrieved WHO passages.")

            st.info(
                "TF-IDF retrieves relevant passages from the WHO document, and Gemini uses those passages to generate answers. "
                "It does not provide diagnosis or personalized medical advice."
            )


# ============================================================
# 4. DATASET ANALYTICS
# ============================================================

elif page == "Dataset Analytics":

    st.title("Dataset Analytics")

    st.caption(
        "Explore the diabetes dataset used in this academic project."
    )

    if not DATA_PATH.exists():

        st.error("Dataset not found: dataset/diabetes.csv")

    else:

        df = load_dataset(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime
        )

        if "Outcome" not in df.columns:

            st.error("The dataset does not contain an Outcome column.")

        else:

            total = len(df)

            positive = int((df["Outcome"] == 1).sum())

            negative = int((df["Outcome"] == 0).sum())

            a, b, c = st.columns(3)

            a.metric("Total Records", total)
            b.metric("Outcome 0 Records", negative)
            c.metric("Outcome 1 Records", positive)

            st.divider()

            st.subheader("Outcome Distribution")

            outcome_counts = (
                df["Outcome"]
                .value_counts()
                .sort_index()
            )

            outcome_counts.index = [
                "Outcome 0",
                "Outcome 1"
            ]

            st.bar_chart(outcome_counts)

            st.subheader("Feature Comparison by Outcome")

            numeric_columns = [
                col
                for col in df.select_dtypes(
                    include=np.number
                ).columns
                if col != "Outcome"
            ]

            selected_feature = st.selectbox(
                "Choose a feature",
                numeric_columns
            )

            means = df.groupby("Outcome")[selected_feature].mean()

            means.index = [
                "Outcome 0",
                "Outcome 1"
            ]

            st.bar_chart(means)

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            st.subheader("Descriptive Statistics")

            st.dataframe(
                df.describe().round(2),
                use_container_width=True
            )

            st.caption(
                "Dataset patterns are descriptive and do not "
                "establish medical causation."
            )


# ============================================================
# 5. BMI CALCULATOR
# ============================================================

elif page == "BMI Calculator":

    st.title("BMI Calculator")

    st.caption(
        "Calculate Body Mass Index using height and weight."
    )

    st.info(
        "BMI is a screening measure. It does not directly "
        "measure body fat or determine an individual's health."
    )

    with st.form("bmi_form"):

        col1, col2 = st.columns(2)

        with col1:

            weight = st.number_input(
                "Weight (kg)",
                min_value=1.0,
                max_value=300.0,
                value=65.0
            )

        with col2:

            height = st.number_input(
                "Height (cm)",
                min_value=50.0,
                max_value=250.0,
                value=165.0
            )

        calculate = st.form_submit_button(
            "Calculate BMI",
            use_container_width=True
        )

    if calculate:

        bmi_value = weight / ((height / 100) ** 2)

        st.metric(
            "Your BMI",
            f"{bmi_value:.1f}"
        )

        if bmi_value < 18.5:

            category = "Underweight"

        elif bmi_value < 25:

            category = "Healthy Weight Range"

        elif bmi_value < 30:

            category = "Overweight"

        else:

            category = "Obesity Range"

        st.subheader(category)

        if bmi_value < 18.5:
            st.warning("⚠️ Underweight")
            st.write(
                "You may benefit from gaining weight under "
                "medical guidance."
            )

        elif bmi_value < 25:
            st.success("✅ Healthy Weight Range")
            st.write(
                "Your BMI falls within the healthy range."
            )

        elif bmi_value < 30:
            st.warning("⚠️ Overweight")
            st.write(
                "Consider discussing your health goals with "
                "a healthcare professional."
            )

        else:
            st.error("🚨 Obesity Range")
            st.write(
                "This BMI range is associated with higher health "
                "risks. Consider consulting a healthcare professional."
            )

        st.info(f"Your BMI is {bmi_value:.1f}")

        st.caption(
            "These are general adult BMI categories, not a diagnosis. "
            "BMI interpretation may differ for children, pregnancy "
            "and some populations."
        )


# ============================================================
# 6. HEALTH EDUCATION
# ============================================================

elif page == "Health Education":

    st.title("Health Education")

    st.caption(
        "General educational information about diabetes."
    )

    topics = {
        "What is diabetes?": (
            "Diabetes is a chronic disease that occurs when "
            "the pancreas does not produce enough insulin or "
            "when the body cannot effectively use the insulin "
            "it produces."
        ),

        "Common symptoms": (
            "Common symptoms can include increased thirst, "
            "frequent urination, tiredness, blurred vision "
            "and unexplained weight changes. Some people may "
            "have few or no noticeable symptoms."
        ),

        "Risk factors": (
            "Risk factors vary by diabetes type. They can include "
            "family history, age, excess body weight, physical "
            "inactivity and other health factors."
        ),

        "Prevention and healthy habits": (
            "Healthy eating, regular physical activity, avoiding "
            "tobacco and maintaining a healthy lifestyle can help "
            "reduce the risk of type 2 diabetes."
        ),

        "When to seek medical advice": (
            "If you have symptoms or concerns about diabetes, "
            "consult a qualified healthcare professional for "
            "appropriate testing and advice."
        )
    }

    for title, content in topics.items():

        with st.expander(title):

            st.write(content)

    st.caption(
        "Educational overview. Source material: WHO diabetes fact sheet."
    )

    st.info(
        "For severe or sudden symptoms, seek urgent medical care."
    )


# ============================================================
# 7. ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title("About MediAssist AI")

    st.caption(
        "AI-Powered Disease Prediction and Medical RAG Assistant"
    )

    st.markdown("""
    ### Project Objective

    This academic prototype demonstrates how machine learning
    and information retrieval can support access to educational
    health information.

    ### Technologies Used

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Logistic Regression
    - TF-IDF and cosine similarity
    - Streamlit
    - WHO diabetes fact sheet

    ### Main Modules

    1. Diabetes prediction using a trained classification model.
    2. Medical document retrieval using TF-IDF.
    3. Dataset exploration and visualization.
    4. BMI calculation.
    5. General diabetes education.

    ### Current Limitations

    The medical assistant retrieves passages from a WHO document.
    It does not yet use a generative language model to compose answers.

    The diabetes model is an educational prototype and is not
    a clinically validated medical device.
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MediAssist AI | Academic Educational Prototype | "
    "Not intended for diagnosis or treatment."
)











