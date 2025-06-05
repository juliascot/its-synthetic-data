'''
This file decomposes and reconstructs the tensor for various ranks, printing the root mean squared error
and accuracy.

Set desired filename, is_stratified, ranks, l2 (within parafac call), and n_splits (within KFold)
'''

import numpy as np
import tensorly as tl
import pandas as pd
from tensorly.decomposition import parafac
from sklearn.model_selection import KFold
from tensor import Tensor

filename = ""
is_stratified = False # Set this to true if we want results to have the data round to zeros or ones (will also print the accuracy)
ranks = range(1, 9)
l2 = 0 # Regularization -- basically to what degree we ignore potential outliers.
n_splits = 30 # The k in k-fold cross-validation
timestamp_cutoff_weight = 1 # Multiply the max timestamp by this to produce the cutoff for guessing whether completed milestones from timestamps
added_timestamp_degree = 1.5 # This is multiplied by the max timestamp to produce the timestamps for unachieved milestones. Used only when adding in new timestamps pre-decomposition


# Helper functions

def stratify_points(tensor): # Makes points 0 or 1 -- only for 3-way tensors
    for student in range(len(tensor)):
        for milestone in range(len(tensor[0])):
            if tensor[student][milestone][2] >= 0.5:
                tensor[student][milestone][2] = 1
            else:
                tensor[student][milestone][2] = 0
    return tensor

def find_accuracy(orig_achieved_slice, reconstructed_achieved_slice, test_indices): # Reports train and test accuracy

    correct_test = 0
    correct_train = 0
    # num_test_points = len(test_indices)
    # num_train_points = len(orig_present_points) - num_test_points

    # for index in range(num_train_points):
    #     tensor_index = orig_present_points[index]
    #     if index in test_indices:
    #         if orig_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]] == constructed_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]]:
    #             correct_test += 1
    #     else:
    #         if orig_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]] == constructed_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]]:
    #             correct_train += 1
    
    # return correct_train / num_train_points, correct_test / num_test_points


def decomp_and_errors(orig_tensor_class: Tensor,
                      ranks: list[int], 
                      train_indices: np.ndarray, 
                      test_indices: np.ndarray, 
                      is_baseline: bool = False, 
                      timestamp_cutoff_weight: float = None, 
                      added_timestamp_degree: float = None
    ) -> tuple[float, float]:
    pass

def collect_all_errors(orig_tensor_class: Tensor, 
                       ranks: list[int], 
                       all_train_indices: np.ndarray, 
                       all_test_indices: np.ndarray, 
                       is_baseline: bool = False, 
                       timestamp_cutoff_weight: float = None, 
                       added_timestamp_degree: float = None,
                       should_print_after: bool = True
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:

    timestamp_train_errors = {rank: [] for rank in ranks}
    timestamp_test_errors = {rank: [] for rank in ranks}
    attempt_train_errors = {rank: [] for rank in ranks}
    attempt_test_errors = {rank: [] for rank in ranks}
    train_accuracy = {rank: [] for rank in ranks}
    test_accuracy = {rank: [] for rank in ranks}




    if should_print_after:
        print(f"  Timestamp Train Errors: {timestamp_train_errors}")
        print(f"  Timestamp Test Errors: {timestamp_test_errors}")
        print(f"  Attempt Train Errors: {attempt_train_errors}")
        print(f"  Attempt Test Errors: {attempt_test_errors}")
        if not is_baseline:
            print(f"  Milestone Attempted Train Accuracy: {train_accuracy}")
            print(f"  Milestone Attempted Test Accuracy: {test_accuracy}")
    pass



if __name__ == "__main__":

    baseline_tensor = Tensor(filename)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_train_indices, all_test_indices = kf.split(baseline_tensor.orig_present_points)

    print("Baseline tensor RMSEs (no modification to identify which milestones are achieved):")
    collect_all_errors(baseline_tensor, ranks, all_train_indices, all_test_indices, is_baseline=True)

    print("Add extra achieved milestone slice to feature dimension, RMSEs and accuracy:")
    collect_all_errors(Tensor(filename, contains_whether_achieved=True), ranks, all_train_indices, all_test_indices)


    for timestamp_cutoff_weight in timestamp_cutoff_weights:
        for added_timestamp_degree in added_timestamp_degrees:
            print(f"RMSEs and Accuracy using timestamp cutoff weight {timestamp_cutoff_weight}, added timestamp degree {added_timestamp_degree}:")
            collect_all_errors(baseline_tensor, ranks, all_train_indices, all_test_indices, timestamp_cutoff_weight=timestamp_cutoff_weight, added_timestamp_degree=added_timestamp_degree)


