from typing import List

from pypdf import PdfReader

import torch
import numpy as np
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer



class TextEmbedding:
    def __init__(self, model_name: str):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def embed(self, docs: List, max_length: int = 8192) -> np.ndarray:
        """
        Embeds list of documents to vector representations

        Args
        ----
        docs: List
            Documents to be embedded
        
        Returns
        -------
        array: np.ndarray
            Numpy array of shape (M, N) representing normalized vector embeddings, where
            M - number of documents in a list; N - dimension of model vectors 
        """
        docs_tokens = self.tokenizer(docs, return_tensors='pt',
                                     padding=True, truncation=True, max_length=max_length)
        docs_embeddings = self.model(**docs_tokens)[0][:, 0]
        normalized_embeddings = nn.functional.normalize(docs_embeddings, p=2.0, dim=1).detach().numpy()
        return normalized_embeddings
    

class PdfChunkReader():
    def __init__(self, pdf_path: str, chunk_length: int = 2500):
        self.pdf_path = pdf_path
        self.chunk_length = chunk_length
        self.pdf_chunks = None
    
    @property
    def pdf_lines(self):
        """
        Returns a generator of all pdf lines 
        """
        pdf_reader = PdfReader(self.pdf_path)
        for page in pdf_reader.pages:
            for line in page.extract_text().split("\n"):
                yield line

    def get_chunks(self):
        if self.pdf_chunks != None:
            return self.pdf_chunks
        
        pdf_chunks = []
        current_lines_list = []
        for line in self.pdf_lines:
            current_lines_list.append(line+"\n")
            if len("".join(current_lines_list)) >= self.chunk_length:
                pdf_chunks.append("".join(current_lines_list[:-1]))
                current_lines_list = [current_lines_list[-1]]
        # Check whether there are leftover lines
        if len(current_lines_list) != 0:
            pdf_chunks.append("".join(current_lines_list).rstrip("\n"))

        self.pdf_chunks = pdf_chunks
        return pdf_chunks


"--------------------------------------------------------------------------------"