# This file is going to be used for terminal frontend for system
import os
import sys
import uuid
import json
import warnings
sys.path.append("..\\")
warnings.filterwarnings("ignore")

from systemComponents import LlmModelModule, RetrieverModule, embedding_model, VectorDb

from transformers import AutoModel, AutoTokenizer

import pypdf
import requests
from pypdf import PdfReader
from sqlalchemy import create_engine, select
from prompts import get_technologies_prompt, get_suggestions_prompt

# GLOBAL VARIABLES
DB_PASSWORD = os.getenv("DB_PASSWORD")
engine = create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost/retriever_chunk_dbs")
MODEL_AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Generate access token for gigachat model using auth token to save it to ACCESS_TOKEN variable
def get_access_token(auth_token):
  url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

  payload = 'scope=GIGACHAT_API_PERS'
  headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
    'RqUID': f'{str(uuid.uuid4())}',
    'Authorization': f'Basic {auth_token}'
  }

  access_response = requests.request("POST", url, headers=headers, data=payload, verify=False)

  ACCCESS = access_response.text
  ACCCESS_TOKEN = json.loads(ACCCESS)["access_token"]
  return ACCCESS_TOKEN

ACCESS_TOKEN = get_access_token(MODEL_AUTH_TOKEN)
CHUNK_LENGTH = 1350 


if __name__ == "__main__":
  model = LlmModelModule(access_token = ACCESS_TOKEN)
  pdf_file_path = input("Input path to pdf file: ")
  pdf_text = ""
  pdf_reader = PdfReader(stream = pdf_file_path)
  for page in pdf_reader.pages:
    page_text = page.extract_text()
    pdf_text = "".join([pdf_text, page_text])
  model_based_technologies = model.generate(get_technologies_prompt(pdf_text))
  vectordb = VectorDb(relational_db_engine = engine,
                      embedding_function = embedding_model,
                      tablename = f'DOCS_CHUNK_SIZE_{CHUNK_LENGTH}')
  vectordb.load_index(f"c:\\main\\GitHub\\documentReviewSystem\\experiments\\vector_dbs\\EMBD_CHUNKS_OF_SIZE_{CHUNK_LENGTH}.index")
  retriever_module = RetrieverModule(vectordb)
  retriver_result = retriever_module.retrieve(model_based_technologies)
  retrieved_docs = [tup[1] for tup in retriver_result]
  print(f"Extracted technologies: {model_based_technologies}")
  print(f"Retrieved docs:\n{retrieved_docs}")






