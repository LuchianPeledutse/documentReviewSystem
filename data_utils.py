from typing import List

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
        normalized_embeddings = nn.functional.normalize(docs_embeddings, p=2.0, dim=1)
        return normalized_embeddings
    

"--------------------------------------------------------------------------------"