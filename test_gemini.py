import streamlit as st
from google import genai

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with: Gemini connection successful!"
    )

    print("SUCCESS!")
    print(response.text)

except Exception as e:
    print("Connection failed.")
    print("Error:", str(e))