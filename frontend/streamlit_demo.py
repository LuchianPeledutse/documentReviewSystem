from typing import List

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/process"  # Replace with actual backend URL

# def extract_text_from_pdf(pdf_file) -> List[str]:
#     """Extract text from each page of PDF, return list of page texts."""
#     reader = PyPDF2.PdfReader(pdf_file)
#     return [page.extract_text() or "" for page in reader.pages]

st.set_page_config(layout="wide")
st.title("Document Review System")

left_col, right_col = st.columns([2, 1])

with left_col:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file is not None:
        st.success("PDF loaded")
        # Optionally display PDF preview (basic)
        st.markdown(f"**File:** {uploaded_file.name}")

        if st.button("Process Document"):
            with st.spinner("Processing ..."):
                pages_text = ["one", "two", "three"] # extract_text_from_pdf(uploaded_file)
                payload = {"pages": pages_text}
                try:
                    # response = requests.post(BACKEND_URL, json=payload)
                    # response.raise_for_status()
                    # result = response.json()
                    result = {"key technologies": "ML", "key technologies from knowledge base documents": "Semantic technologies", "suggested recommendations": "Something from gigachat model"}
                except Exception as e:
                    st.error(f"Backend error: {e}")
                    st.stop()

            with right_col:
                st.subheader("Results")
                st.markdown("**Key Technologies**")
                st.write(result.get("key technologies", "N/A"))
                st.markdown("**Key Technologies from Knowledge Base**")
                st.write(result.get("key technologies from knowledge base documents", "N/A"))
                st.markdown("**Suggested Recommendations**")
                st.write(result.get("suggested recommendations", "N/A"))