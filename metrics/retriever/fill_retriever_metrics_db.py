import os
import sys
import pickle
import pathlib
sys.path.append("C:\\main\\GitHub\\documentReviewSystem")

from sqlalchemy import create_engine

from data_utils import chunk_pdf_file_pages
from systemComponents import VectorDb, embedding_model

# GLOBAL
DB_PASSWORD = os.getenv("DB_PASSWORD")
ENGINE = create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost/retriever_chunk_dbs")
CHUNK_SIZES = [150*i for i in range(1, 13)]
LOG_NAMES = []

with open("log_files_chunked.txt", 'r') as log_files:
    for line in log_files:
        LOG_NAMES.append(line.strip())

if len(LOG_NAMES) != 1:
    print("UPDATING IS NOT IMPLEMENTED. DELETE ALL LOGS AND RETRY.")

# Loading dict that represents labels
with open("label_dict.pkl", "rb") as label_dict_file:
    label_dict = pickle.load(label_dict_file)

# Preparing paths to iterate over
retriever_metrics_folder = pathlib.Path(".")
dir_paths = [item for item in retriever_metrics_folder.iterdir() if item.is_dir()]


# For each chunk create a separate database
for chunk_size in CHUNK_SIZES:
    # Create vector database to embed chunks
    vectordb = VectorDb(relational_db_engine = ENGINE,
                        embedding_function = embedding_model,
                        tablename = f"DOCS_CHUNK_SIZE_{chunk_size}")
    vectordb.create_index()
    for dir in dir_paths:
        label = label_dict[dir.name]
        for doc in dir.iterdir():
            # Keep track of files that were chunked (for future UPDATE IMPLEMENTATION)
            if doc.name not in LOG_NAMES:
                with open("log_files_chunked.txt", "a") as log_files:
                    log_files.write(doc.name + "\n")
            pages_chunks = chunk_pdf_file_pages(dir.name + '\\' + doc.name, chunk_length = chunk_size)
            for chunks in pages_chunks:
                vectordb.load_chunks(chunks, label = label)
    vectordb.save_index(f".\\EMBD_CHUNKS_OF_SIZE_{chunk_size}.index")

