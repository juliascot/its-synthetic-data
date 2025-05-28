import sys
import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from scipy.special import expit
from scipy.optimize import curve_fit
from internal_imports.tensor import Tensor
from internal_imports.learning_curve_analysis import extract_prior_and_acquired_knowledge



################################
#       HYPERPARAMETERS        #
################################

l2 = 0
filename = "Getting_Started_Reordered.csv"
rank = 6





################################
#      HELPER FUNCTIONS        #
################################

def power_law(x, a, b):
    return a * np.power(x, b)



################################
#             MAIN             #
################################

if __name__ == "__main__":

    # Decompose and reconstruct the tensor
    initial_tensor = Tensor(filename)

    mask = ~np.isnan(initial_tensor.data_tensor)
    initial_tensor.data_tensor = np.nan_to_num(initial_tensor.data_tensor)

    weights, factors = parafac(initial_tensor.data_tensor, rank=rank, mask=mask, l2_reg=l2)
    reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

    all_extracted_info = extract_prior_and_acquired_knowledge(reconstructed_tensor)
        
    # Graph each question
    for question_number, question in enumerate(all_extracted_info):

        plt.figure()
        plt.scatter(question[0], question[1], label='Data', c="blue")

        plt.suptitle(f'Question {question_number + 1} Learning Curve',fontsize=16, y=0.97)
        plt.title(f'Across all students, Rank = {rank}, L2 = {l2}', fontsize=8)
        plt.xlabel("$\t{a}$: prior knowledge")
        plt.ylabel("$\t{b}$: learning rate")

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

