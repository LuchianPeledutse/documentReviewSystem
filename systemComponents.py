import os
import uuid
import json
from typing import List, Tuple, Callable

from data_utils import normalize_vectors

import requests

import torch
import numpy as np

from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL_NAME = 'Snowflake/snowflake-arctic-embed-l-v2.0'

def embedding_model(docs: List[str], device: str = 'cuda') -> np.ndarray:
    """
    Given a list of documents returns their respective normalized embeddings
    Model used is hardcoded. In future might be softened with ONNX
    """
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME,
                                      add_pooling_layer = False).to(device = device)
    model.eval()
    tokenized_queries = tokenizer(["query: " + doc for doc in docs],
                                  padding=True, truncation=True,
                                  return_tensors='pt', max_length=8192).to(device = 'cuda')
    with torch.no_grad():
        query_embeddings = model(**tokenized_queries)[0][:, 0]

    normalized_embeddings = normalize_vectors(query_embeddings.detach().cpu().numpy())
    return normalized_embeddings

    

'--------------------------------------------------------------------------------'

class LlmModelModule:
    def __init__(self, access_token: str):
        self.access_token = access_token

    def generate(self, prompt: str) -> str:
        """Given a prompt generates response using the model API"""
        rq_uid = str(uuid.uuid4())
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        payload = json.dumps({
        "model": "GigaChat",
        "messages": [
            {
            "role": "user",
            "content": prompt,
            }
        ],
        "temperature": 1,
        "top_p": 0.1,
        "stream": False,
        "n": 1,
        "max_tokens": 2048,
        "repetition_penalty": 1,
        "update_interval": 0
        })
        headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        "RqUID": rq_uid,
        'Authorization': f'Bearer {self.access_token}'
        }

        response = requests.request("POST", url,
                                    headers=headers,
                                    data=payload, verify=False)
        return json.loads(response.text)["choices"][0]["message"]["content"]

class VectorDb:
    """
    Implementation of vector database (vector index with additional operations)
    """
    def __init__(self):
        self.vector_index = None

    def load_database(self, load_path: str) -> None:
        "Loads database given path"
        pass

    def save_database(self, save_path: str) -> None:
        "Saves database given path"
        pass

class RetrieverModule:
    def __init__(self, embedding_model: Callable[[List[str]], np.ndarray],
                 vector_database: VectorDb, relational_db_engine: str):
        self.embedding_model = embedding_model
        self.vector_database = vector_database
        self.relational_db_engine = relational_db_engine

    def retrieve(self, text: str) -> List[Tuple[int, str]]:
        """
        Given text retrieves the indicies from faiss database and chunks from postgres database
        Each index relates to its chunk, respectively
        """ 
        pass

class AugmentationModule:
    def __init__(self, prompt: str):
        pass

"--------------------------------------------------------------------------------"