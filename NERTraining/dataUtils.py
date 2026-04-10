import re
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

import fitz
import torch
from torch.utils.data import Dataset, DataLoader    

from transformers import AutoModel, AutoTokenizer


def pages_with_annotations(folder_path: str = ".\\TrainData") -> List[Tuple[str, Tuple[Tuple[int, int], ...] | Tuple[()]]]:
    """
    Given path to folder with pdf files returns text with pages and spans of annotated entities

    Returns
    -------
    annotated_chunks: List[Tuple[str, Tuple[Tuple[int, int], ...]]]
        List that contains tuples that contain page chunks and tuples of entity spans
        Example: [("These are John and Bobby", ((10, 14), (19, 24))), ("Hi", None)]
    """
    pdf_paths = list(Path(folder_path).iterdir())
    data_list = []
    for pdf_path in tqdm(pdf_paths, desc="Going through pdf documents"):
        with fitz.open(pdf_path) as pdf_doc:
            for page in range(len(pdf_doc)):
                spans = []
                current_page = pdf_doc[page]
                current_text = current_page.get_text()
                annotations_rectangles = list(annot.rect for annot in current_page.annots())
                for rect in annotations_rectangles:
                    # Find entity text to find its span
                    entity_text = current_page.get_textbox(rect)
                    # Find that span to add to data unit
                    entity_span = re.search(re.escape(entity_text), current_text).span()
                    spans.append(entity_span)
                data_list.append((current_text, tuple(spans)))
    return data_list

"--------------------------------------------------------------------------------"