import os
import sys
from typing import List, Callable

sys.path.extend(["..\\", "..\\backend\\"])

import pypdf
import faiss

from tqdm import tqdm

from data_utils import TextEmbedding, PdfChunkReader
from relational_database import insert_chunks

PDF_DOCS_PATH = None
VECTOR_DATABASE_PATH = None

class FaissLoader:
    def __init__(self, embedding_class: TextEmbedding):
        self.embedding_class = embedding_class

    def load_and_save_database(self, folder_path: str, save_to: str, db_chunk_add: Callable[[List[str], List[int]], None]) -> None:
        """
        Loads all documents from folder path that contains only pdf docs
        chunks them and embeds into vector database (embedding vectors are normalized)
        and saves it to save_to path. It also adds chunks to database with specified function
        db_chunk_add
        """
        index = faiss.IndexFlatIP(self.embedding_class.model.embeddings.word_embeddings.weight.shape[1])
        pdf_filenames = os.listdir(folder_path)
        CURRENT_TOTAL_VECTORS = 0
        for pdf_filename in tqdm(pdf_filenames, desc="Going through filenames..."):
            chunks = PdfChunkReader(folder_path + "\\" + pdf_filename).get_chunks()
            embedded_chunks_yielding = self.embedding_class.embed(chunks)
            previous_total = CURRENT_TOTAL_VECTORS
            for chunk_batch in embedded_chunks_yielding:
                index.add(chunk_batch)
                next_total = index.ntotal
                db_chunk_add(chunks[previous_total:next_total], [id for id in range(previous_total, next_total)])
                previous_total = next_total
            CURRENT_TOTAL_VECTORS = previous_total
        
        faiss.write_index(index, save_to)
        return None

if __name__ == "__main__":
    embedding_class = TextEmbedding(model_name='Snowflake/snowflake-arctic-embed-l-v2.0', device="cpu")
    faiss_loader = FaissLoader(embedding_class=embedding_class)
    faiss_loader.load_and_save_database(folder_path= "C:\\main\\GitHub\\documentReviewSystem\\knowledge_data",
                                        save_to= "C:\\main\\GitHub\\documentReviewSystem\\project_data\\vector_db.index",
                                        db_chunk_add=insert_chunks)
