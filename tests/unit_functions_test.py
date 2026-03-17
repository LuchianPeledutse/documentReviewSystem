import os
import sys
import random
import string

random.seed(42)
sys.path.append("C:\\main\\GitHub\\documentReviewSystem")

import pypdf
import pytest
import numpy as np

from systemComponents import embedding_model
from data_utils import normalize_vectors, chunk_pdf_file_pages

# GLOBAL VARIABLES
TEST_PDF_FOLDER_PATH = "C:\\main\\GitHub\\documentReviewSystem\\data_test_pdf_chunks"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# VECTOR NORMALIZATION TEST
@pytest.mark.parametrize("vector_matrix",
                         [rng.random((10, random.randint(1, 101))) for _ in range(5_000)])
def test_vector_normalization(vector_matrix, epsilon = 1e-10) -> np.array:
    """Test whether vectors ara properly normalized with normalize function"""
    normalized_matrix = normalize_vectors(vector_matrix)
    N, E = vector_matrix.shape
    zero_matrix = np.zeros((N, E), dtype = vector_matrix.dtype)
    for row in range(N):
        single_vector = vector_matrix[row, :]
        norm = np.sqrt(sum([value**2 for value in single_vector]))
        norm_single_vector = single_vector/norm
        zero_matrix[row, :] = norm_single_vector
    assert bool((abs(normalized_matrix - zero_matrix) < epsilon).all())

# EMBEDDING FUNCTION OUTPUT NORMALIZATION TEST
@pytest.fixture()
def random_string_batch(num_batches: int = 10, string_length: int = 200):
    string_batch = []
    for _ in range(num_batches):
        letters = string.ascii_letters + string.punctuation
        random_string = ''.join(random.choices(letters, k=string_length))
        string_batch.append(random_string)
    return string_batch

def test_embedding_normalization(random_string_batch, epsilon: float = 1e-5):
    """
    Tests whether the vectors returned by embedding model are normalized
    """
    normalized_embeddings = embedding_model(random_string_batch)
    assert bool(abs(normalized_embeddings-np.ones(normalized_embeddings.shape)).all())


# PDF CHUNKER TEST
@pytest.fixture()
def pdf_files_names_list():
    pdf_files_names = os.listdir(TEST_PDF_FOLDER_PATH)
    return pdf_files_names

@pytest.mark.parametrize("chunk_length", 
                         rng.integers(low = 50, high = 2000, size = 10).tolist())
def test_pdf_chunker(pdf_files_names_list, chunk_length: int) -> None:
    "This function checks pdf chunker function from data_utils file"
    for pdf_name in pdf_files_names_list:
        full_pdf_path = TEST_PDF_FOLDER_PATH + f"\\{pdf_name}"
        # Extract text from each pdf page and compare that text to joined chunks
        pdf_reader = pypdf.PdfReader(full_pdf_path)
        pdf_pages_chunks = chunk_pdf_file_pages(pdf_path = full_pdf_path,
                                                chunk_length = chunk_length)
        # Iterate through pages and compare their text to joined chunk text
        current_page_idx = 0
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            assert page_text == "".join(pdf_pages_chunks[current_page_idx]), f"Error on page {current_page_idx}"
            current_page_idx += 1