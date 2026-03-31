import os
import sys
import json
import uuid
import rouge
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pypdf import PdfReader
from rouge_score import rouge_scorer
sys.path.append('..\\..\\')

from sqlalchemy import create_engine, select
from prompts import get_technologies_prompt, get_suggestions_prompt
from systemComponents import LlmModelModule, RetrieverModule, embedding_model, VectorDb

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
# Faithfullness with gigachat technology extraction 
model = LlmModelModule(access_token = ACCESS_TOKEN)
vectordb = VectorDb(relational_db_engine = engine,
                    embedding_function = embedding_model,
                    tablename = f'DOCS_CHUNK_SIZE_{CHUNK_LENGTH}')
vectordb.load_index("c:\\main\\GitHub\\documentReviewSystem\\experiments\\vector_dbs\\EMBD_CHUNKS_OF_SIZE_1350.index")
# Accumulate pdf paths (input documents)
path = Path(".")
pdf_paths = []
for item in path.iterdir():
  if item.is_dir():
    for pdf_path in item.iterdir():
      pdf_paths.append(pdf_path)
      
current_score = None
for pdf_file_path in tqdm(pdf_paths, desc="Calculating average ROUGE metric"):
    pdf_text = ""
    pdf_reader = PdfReader(stream = pdf_file_path)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        pdf_text = "".join([pdf_text, page_text])
    reference_technologies = model.generate(get_technologies_prompt(pdf_text))
    retriever_module = RetrieverModule(vectordb)
    retriver_result = retriever_module.retrieve(reference_technologies)
    retrieved_docs = [tup[1] for tup in retriver_result]
    generated_technologies = model.generate(get_suggestions_prompt(reference_technologies, "".join(retrieved_docs)))
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score("".join(retrieved_docs), generated_technologies)["rouge1"]
    if current_score is None:
       current_score = np.array(scores)
    else:
       current_score += np.array(scores)
print(f"AVG ROUGE: {current_score/len(pdf_paths)}")