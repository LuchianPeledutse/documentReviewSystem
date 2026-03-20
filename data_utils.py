from typing import List

from pypdf import PdfReader

import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

# CLASSES
# -------

class TextEmbedding:
    # Given a device and model name implements a method for embedding documents
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def embed(self, docs: List, max_length: int = 8192, batch_size: int = 3) -> np.ndarray:
        """
        Embeds list of documents to vector representations by yilding embedded batches

        Yields
        ------
        array: np.ndarray
            Numpy array of shape (B, E) representing normalized vector embeddings, where
            B - number of documents in a batch; E - dimension of model vectors 
        """
        NUM_DOCS = len(docs)
        for i_batch in range(1, NUM_DOCS//batch_size + 1):
            docs_tokens = self.tokenizer(docs[(i_batch-1)*batch_size:i_batch*batch_size if i_batch*batch_size <= NUM_DOCS else NUM_DOCS],
                                         return_tensors='pt', padding=True, truncation=True, max_length=max_length)

            docs_device_tokens = {k: v.to(device=self.device) for k, v in docs_tokens.items()}
            docs_embeddings = self.model(**docs_device_tokens)[0][:, 0]
            normalized_embeddings = nn.functional.normalize(docs_embeddings, p=2.0, dim=1).detach().cpu().numpy()
            yield normalized_embeddings



# FUNCTIONS
# ---------

def chunk_pdf_file_pages(pdf_path: str, chunk_length: str) -> List[List]:
    """
    Given pdf path and chunk_length splits pdf pages to chunks of length
    approximate to chunk_length

    Returns
    -------
    chunks_of_pages: List[List]
        List containing lists of chunks for pages in order relative to pdf pages order
    """
    # Reads pdf file into memory to iterate over pages
    pdf_reader = PdfReader(stream = pdf_path)
    # Iterate over pages to chunk each page separately
    chunks_of_pages = []
    for page in tqdm(pdf_reader.pages, desc="Going through pdf pages..."):
        page_text = page.extract_text()
        page_text_lines = page_text.split("\n")
        page_chunks = []
        current_line_list = []
        # Iterate over lines of a single page to make chunks
        for line in page_text_lines:
            current_line_list.append(line + "\n")
            chunk_candidate = "".join(current_line_list)
            if len(chunk_candidate) >= chunk_length:
                page_chunks.append("".join(current_line_list[:-1]))
                current_line_list = [current_line_list[-1]]
        if len(current_line_list) != 0:
            page_chunks.append("".join(current_line_list))
        # Copy last chunk from the page to make chunk joining consistent
        last_chunk_length = len(page_chunks[-1])
        last_chunk_from_page = page_text[-last_chunk_length:].lstrip()
        page_chunks[-1] = last_chunk_from_page
        # Remember chunk pages 
        chunks_of_pages.append(page_chunks)
    return chunks_of_pages


def normalize_vectors(vectors_array: np.ndarray) -> np.ndarray:
    """
    Function that takes in numpy vectors as rows, normalizes and returns them back

    Args
    ----
    vectors_array: np.ndarray
        Numpy array of shape N x E where N is number of vectors and E is their dimension

    Returns
    -------
    normalized_vectors: np.ndarray
        Same vectors but normalized 
    """
    norms = np.apply_along_axis(np.linalg.norm, arr = vectors_array, axis = 1) # Shape N 
    normalized_vectors = vectors_array/norms[:, np.newaxis]
    return normalized_vectors