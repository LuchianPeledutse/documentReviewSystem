import os
import sys
from io import BytesIO
sys.path.append("..\\")

from typing import List
from pypdf import PdfReader
from data_utils import extract_pages_from_pdf

import requests
import streamlit as st

LLM_URL = os.getenv("LLM_URL")
BACKEND_URL = os.getenv("BACKEND_URL")
NER_model_URL = os.getenv("NER_model_URL")

st.set_page_config(layout="wide")
st.title("Система аудита учебных материалов")

left_col, right_col = st.columns([2, 1])

with left_col:
    uploaded_file = st.file_uploader("Загрузить pdf", type="pdf")
    if uploaded_file is not None:
        st.success("Файл загружен")
        # Optionally display PDF preview (basic)
        st.markdown(f"**PDF файл:** {uploaded_file.name}")
        if st.button("Обработать документ"):
            with st.spinner("Обработка документа ..."):
                    page_texts = extract_pages_from_pdf(BytesIO(uploaded_file.getvalue()))
                    try:
                        # Collect technologies extracted from the documents provided
                        extracted_doc_technologies = []
                        for page_text in page_texts:
                             # Here we employ the NER model
                             current_page_technologies = requests.post(url = "http://127.0.0.1:8002/entities", json = {"text": page_text})
                             extracted_doc_technologies.extend(current_page_technologies.json()["entities"])
                        retrieved_model = requests.post(url = "http://127.0.0.1:8001/chunks", 
                                                        json = {"text": "NLP, ML, LLMs, retriever, matrix operations, Spline methods"})
                        retrieved_chunks = retrieved_model.json()["chunks"]
                        # Collect technologies extracted from the chunks provided
                        extracted_knowledge_base_technologies = []
                        for chunk in retrieved_chunks:
                            # Here we employ NER model
                            current_page_technologies = requests.post(url = "http://127.0.0.1:8002/entities", json = {"text": chunk})
                            extracted_knowledge_base_technologies.extend(current_page_technologies.json()["entities"])
                        all_technologies = extracted_doc_technologies + extracted_knowledge_base_technologies
                        llm_suggestions = requests.post("http://127.0.0.1:8000/suggestions",
                                                        json = {"string_technologies": " ".join(all_technologies)})
                        result = {"document_technologies": " ".join(extracted_doc_technologies),
                                  "knowledge_base_technologies": " ".join(extracted_knowledge_base_technologies),
                                  "model_recommendations": llm_suggestions.json()["answer"]}
                    except Exception as e:
                        st.error(f"Backend error: {e}")
                        st.stop()

            with right_col:
                st.subheader("Результаты")
                st.markdown("**Технологии извлеченные из документа**")
                st.write(result.get("document_technologies", "N/A"))
                st.markdown("**Технологии извлеченные из внутренних документов**")
                st.write(result.get("knowledge_base_technologies", "N/A"))
                st.markdown("**Рекомандации на основе параметрических знаний модели**")
                st.write(result.get("model_recommendations", "N/A"))

"--------------------------------------------------------------------------------"