import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from tensor import Tensor

filename = "Getting_Started_Processed.csv"
rank = 8
l2 = 0

def create_dense_tensor(filename: str, rank: int, l2: float) -> np.ndarray:
    # Decompose and reconstruct the tensor
    initial_tensor = Tensor(filename, is_student_outside=True)

    mask = ~np.isnan(initial_tensor.data_tensor)
    initial_tensor.data_tensor = np.nan_to_num(initial_tensor.data_tensor)

    weights, factors = parafac(initial_tensor.data_tensor, rank=rank, mask=mask, l2_reg=l2)
    return tl.kruskal_to_tensor((weights, factors))




