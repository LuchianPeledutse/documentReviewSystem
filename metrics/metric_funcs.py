from typing import List
import numpy as np


def precision(values: List[int], true_label: int) -> float:
    np_values = np.array(values, dtype=np.int32) if type(values) != np.ndarray else values
    return ((np_values == true_label).sum()/len(np_values)).item()

def average_precision(values: List[int], true_label: int) -> float:
    np_values = np.array(values, dtype=np.int32) if type(values) != np.ndarray else values
    ranks = (np.where(np_values == true_label)[0] + 1).tolist()
    return 1/len(np_values)*sum(precision(np_values[:rank_i], true_label) for rank_i in ranks)

def rr(values: List[int], true_label: int) -> float:
    np_values = np.array(values, dtype=np.int32) if type(values) != np.ndarray else values
    first_rank = (np.where(np_values == true_label)[0] + 1).tolist()[0]
    return 1/first_rank