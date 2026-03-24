import os
import re
import uuid
import json
import warnings
from typing import List, Tuple, Dict, Callable

warnings.filterwarnings("ignore")

import requests
from tqdm import tqdm

from data_utils import normalize_vectors

import faiss
import torch
import numpy as np

from sqlalchemy import Text, Integer, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session, DeclarativeBase, Mapped, mapped_column

from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL_NAME = 'Snowflake/snowflake-arctic-embed-l-v2.0'
MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, add_pooling_layer = False).to(device = 'cuda')
TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

def embedding_model(docs: List[str], device: str = 'cuda') -> np.ndarray:
    """
    Given a list of documents returns their respective normalized embeddings
    Model used is hardcoded. In future might be softened with ONNX
    """
    tokenizer = TOKENIZER
    model = MODEL
    model.eval()
    tokenized_queries = tokenizer(docs,
                                  padding=True, truncation=True,
                                  return_tensors='pt', max_length=8192).to(device = 'cuda')
    with torch.no_grad():
        query_embeddings = model(**tokenized_queries)[0][:, 0]

    normalized_embeddings = normalize_vectors(query_embeddings.contiguous().detach().cpu().numpy())
    return normalized_embeddings

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
    def __init__(self, relational_db_engine: Engine, tablename: str, 
                 embedding_function: Callable[[List[str]], np.ndarray]):
        self.vector_index = None
        self.embedding_function = embedding_function
        self.relational_db_engine = relational_db_engine
        # Create table with specific name
        class Base(DeclarativeBase):
            pass
        class Chunk(Base):
            __tablename__ = tablename

            id: Mapped[int] = mapped_column(primary_key = True)
            content: Mapped[str]
            label: Mapped[int]
        # Create table in postgres database and save it as a model
        Base.metadata.create_all(relational_db_engine)
        self.Chunk = Chunk

    def create_index(self, dim: int = 1024) -> None:
        """
        Builds inner product index given the number of dimensions
        """
        self.vector_index = faiss.IndexFlatIP(dim)

    def load_index(self, load_path: str) -> None:
        "Loads database given path"
        self.vector_index = faiss.read_index(load_path)
        print(f"Vector index read from the path {load_path}")

    def save_index(self, save_path: str) -> None:
        "Saves database given path"
        faiss.write_index(self.vector_index, save_path)
        print(f"Vector index saved at {save_path}")

    def load_chunks(self, chunks: List[str], label: int, batch_size: int = 3) -> None:
        """
        Embeds chunks and loads them to faiss vector database
        Chunks, their label, and corresponding ids are saved to relational database
        """
        assert self.vector_index is not None, "Vector index has to be initialized"
        chunks_without_null = filter(lambda item: not bool(re.findall(r"\x00", item)), chunks)
        chunks = list(map(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8'), chunks_without_null))
        N_total = len(chunks)
        print(f"Length of chunks: {N_total}")
        print("".join(["-" for  _ in range(1000)]))
        print(chunks)
        print("".join(["-" for  _ in range(1000)]))
        VI_total = self.vector_index.ntotal
        # Load embedded chunks to vector index
        for i in tqdm(range((N_total//batch_size + 1) if N_total%batch_size != 0 else N_total//batch_size),
                      desc=f"Loading chunks to database with batch size {batch_size}"):
            embedded_batch = self.embedding_function(chunks[i*batch_size: ((i+1)*batch_size if (i+1)*batch_size <= N_total else N_total)])
            self.vector_index.add(embedded_batch)
        # load chunks with their corresponding ids and labels
        with Session(self.relational_db_engine) as session:
            chunk_models = [
                self.Chunk(id=VI_total + chunk_idx, content=chunks[chunk_idx], label=label)
                for chunk_idx in range(N_total)
            ]
            session.add_all(chunk_models)
            session.commit()
        
        
        
    
class RetrieverModule:
    def __init__(self, vector_database: VectorDb, k: int = 10):
        self.vector_database = vector_database
        self.k = k
    
    @property
    def vector_index(self):
        return self.vector_database.vector_index

    def retrieve(self, text: str) -> List[Tuple[int, str]]:
        """
        Given text retrieves the indicies from faiss database and chunks from postgres database
        Each index relates to its chunk, respectively
        """ 
        vector_text = self.vector_database.embedding_function(["query: " + text],
                                                              device = "cpu")
        search_result = self.vector_index.search(vector_text, self.k)
        closest_rows = []
        for idx in search_result[1].flatten().tolist():
            statement = select(self.vector_database.Chunk).where(self.vector_database.Chunk.id == idx)
            with Session(self.vector_database.relational_db_engine) as session:
                selected_row = session.execute(statement).all()
                closest_rows.append((selected_row[0][0].id, selected_row[0][0].content, selected_row[0][0].label))
        return closest_rows

class AugmentationModule:
    def __init__(self, prompt: str):
        pass

"--------------------------------------------------------------------------------"