import os
import sys
from typing import List
sys.path.append("..\\")

from systemComponents import RetrieverModule, VectorDb, embedding_model

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class ChunkRequest(BaseModel):
    text: str

class ChunkResponse(BaseModel):
    chunks: List[str]

DB_PASSWORD: str = os.getenv("DB_PASSWORD")
FAISS_VECTOR_DB_PATH: str = os.getenv("FAISS_VECTOR_DB_PATH")
engine = create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost/retriever_chunk_dbs")


# Preparing vector database
vectordb = VectorDb(relational_db_engine = engine,
                    embedding_function = embedding_model,
                    tablename = 'DOCS_CHUNK_SIZE_1350')
vectordb.load_index(FAISS_VECTOR_DB_PATH)
# Preparing retriever module
retriever = RetrieverModule(vector_database = vectordb, k = 10)

app = FastAPI()

@app.post("/chunks")
def get_chunks(chunk_request: ChunkRequest) -> ChunkResponse:
    retrieved_objects = retriever.retrieve(chunk_request.text)
    retrieved_texts = [obj[1] for obj in retrieved_objects]
    return {"chunks": retrieved_texts}

if __name__ == "__main__":
    uvicorn.run(app, port = 8001)