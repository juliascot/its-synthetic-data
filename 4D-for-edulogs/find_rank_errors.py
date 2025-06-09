'''
This file decomposes and reconstructs the tensor for various ranks, printing the root mean squared error
and accuracy.

Set desired filename, is_stratified, ranks, l2 (within parafac call), and n_splits (within KFold)
'''

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.model_selection import KFold
from tensor import Tensor

filename = "parsing/file_wrangler_third_pass_results.csv"
is_stratified = False # Set this to true if we want results to have the data round to zeros or ones (will also print the accuracy)
ranks = range(1, 9)
l2 = 0 # Regularization -- basically to what degree we ignore potential outliers.
n_splits = 30 # The k in k-fold cross-validation
timestamp_cutoff_weights = [0.9, 1, 1.1] # Multiply the max timestamp by this to produce the cutoff for guessing whether completed milestones from timestamps
added_timestamp_degrees = [1.2, 1.5, 2] # This is multiplied by the max timestamp to produce the timestamps for unachieved milestones. Used only when adding in new timestamps pre-decomposition


# Helper functions

def stratify_points(tensor): # Makes points 0 or 1 -- only for 3-way tensors
    for student in range(len(tensor)):
        for milestone in range(len(tensor[0])):
            if tensor[student][milestone][2] >= 0.5:
                tensor[student][milestone][2] = 1
            else:
                tensor[student][milestone][2] = 0
    return tensor


def find_accuracy(orig_achieved_slice: np.ndarray, reconstructed_achieved_slice: np.ndarray): # Reports train and test accuracy

    num_correct = 0

    for i in range(len(orig_achieved_slice)):
        for j in range(len(orig_achieved_slice[0])):
            if orig_achieved_slice[i][j] == reconstructed_achieved_slice[i][j]:
                num_correct += 1
    
    return num_correct/orig_achieved_slice.size


def add_extreme_timestamps(tensor: np.ndarray, extreme_timestamp: float) -> np.ndarray:
    for student in tensor:
        for milestone in student:
            if np.isnan(milestone[0]):
                milestone[0] = extreme_timestamp
    return tensor


def generate_completed_milestone_values_slice(tensor: np.ndarray, max_time: float) -> np.ndarray:
    completed_milestone_slice = np.full((len(tensor), len(tensor[0])), 0)
    for s in range(len(tensor)):
        for m in range(len(tensor[0])):
            if tensor[s][m][0] <= max_time:
                completed_milestone_slice[s][m] = 1
    return completed_milestone_slice


def find_completed_milestones(tensor: np.ndarray) -> np.ndarray: # only used when in third case (use timestamps to guess whether milestone was completed)
    completed_milestone_slice = np.full((len(tensor), len(tensor[0])), 0)
    for s in range(len(tensor)):
        for m in range(len(tensor[0])):
            if not np.isnan(tensor[s][m][0]):
                completed_milestone_slice[s][m] = 1
    return completed_milestone_slice



def decomp_and_errors(orig_tensor_class: Tensor,
                      ranks: list[int], 
                      train_indices: np.ndarray, 
                      test_indices: np.ndarray, 
                      is_baseline: bool = False, 
                      timestamp_cutoff_weight: float = None, 
                      added_timestamp_degree: float = None
    ) -> tuple[dict[int: float], dict[int: float], dict[int: float], dict[int: float], dict[int: float], dict[int: float]]:

    timestamp_train_errors = {rank: None for rank in ranks}
    timestamp_test_errors = {rank: None for rank in ranks}
    attempt_train_errors = {rank: None for rank in ranks}
    attempt_test_errors = {rank: None for rank in ranks}
    train_accuracies = {rank: None for rank in ranks}
    test_accuracies = {rank: None for rank in ranks}

    train_tensor = np.copy(orig_tensor_class.data_tensor)

    if timestamp_cutoff_weight is not None:
        orig_milestones_completed = find_completed_milestones(train_tensor)
        train_tensor = add_extreme_timestamps(train_tensor, added_timestamp_degree * orig_tensor_class.max_time)

    for test_index in test_indices:
        tensor_test_index = orig_tensor_class.orig_present_points[test_index]
        train_tensor[tensor_test_index[0], tensor_test_index[1], :] = np.nan

    mask = ~np.isnan(train_tensor)
    train_tensor = np.nan_to_num(train_tensor)

    for rank in ranks:

        weights, factors = parafac(train_tensor, rank=rank, mask=mask, l2_reg=l2)
        reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

        if not is_baseline:
            if timestamp_cutoff_weight is not None:
                completed_milestone_slice = generate_completed_milestone_values_slice(reconstructed_tensor, orig_tensor_class.max_time)
                train_accuracy, test_accuracy = find_accuracy(orig_milestones_completed, completed_milestone_slice)
            else:
                reconstructed_tensor = stratify_points(reconstructed_tensor)
                train_accuracy, test_accuracy = find_accuracy(orig_tensor_class.data_tensor[:, :, 2], reconstructed_tensor[:, :, 2])

            train_accuracies[rank] = train_accuracy
            test_accuracies[rank] = test_accuracy

        timestamp_mse_train_values, timestamp_mse_test_values, attempt_mse_train_values, attempt_mse_test_values = [], [], [], []
        for train_index in train_indices:
            tensor_train_index = orig_tensor_class.orig_present_points[train_index]
            timestamp_mse_train_values.append((orig_tensor_class.data_tensor[tensor_train_index[0]][tensor_train_index[1]][0] - reconstructed_tensor[tensor_train_index[0]][tensor_train_index[1]][0]) ** 2)
            attempt_mse_train_values.append((orig_tensor_class.data_tensor[tensor_train_index[0]][tensor_train_index[1]][1] - reconstructed_tensor[tensor_train_index[0]][tensor_train_index[1]][1]) ** 2)
        for test_index in test_indices:
            tensor_test_index = orig_tensor_class.orig_present_points[test_index]
            timestamp_mse_test_values.append((orig_tensor_class.data_tensor[tensor_test_index[0]][tensor_test_index[1]][0] - reconstructed_tensor[tensor_test_index[0]][tensor_test_index[1]][0]) ** 2)
            attempt_mse_test_values.append((orig_tensor_class.data_tensor[tensor_test_index[0]][tensor_test_index[1]][1] - reconstructed_tensor[tensor_test_index[0]][tensor_test_index[1]][1]) ** 2)
        timestamp_train_errors[rank] = np.mean(timestamp_mse_train_values)**0.5
        timestamp_test_errors[rank] = np.mean(timestamp_mse_test_values)**0.5
        attempt_train_errors[rank] = np.mean(attempt_mse_train_values)**0.5
        attempt_test_errors[rank] = np.mean(attempt_mse_test_values)**0.5

    return timestamp_train_errors, timestamp_test_errors, attempt_train_errors, attempt_test_errors, train_accuracies, test_accuracies



def collect_all_errors(orig_tensor_class: Tensor, 
                       ranks: list[int], 
                       all_train_indices: np.ndarray, 
                       all_test_indices: np.ndarray, 
                       n_splits: int,
                       is_baseline: bool = False, 
                       timestamp_cutoff_weight: float = None, 
                       added_timestamp_degree: float = None,
                       should_print_after: bool = True
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:

    ranks = [rank for rank in ranks if rank <= len(orig_tensor_class.data_tensor[0][0])]

    all_timestamp_train_errors = {rank: 0 for rank in ranks}
    all_timestamp_test_errors = {rank: 0 for rank in ranks}
    all_attempt_train_errors = {rank: 0 for rank in ranks}
    all_attempt_test_errors = {rank: 0 for rank in ranks}
    all_train_accuracies = {rank: 0 for rank in ranks}
    all_test_accuracies = {rank: 0 for rank in ranks}

    for i in range(n_splits):
        timestamp_train_errors, timestamp_test_errors, attempt_train_errors, attempt_test_errors, train_accuracies, test_accuracies = decomp_and_errors(orig_tensor_class, ranks, all_train_indices[i], all_test_indices[i], is_baseline=is_baseline, timestamp_cutoff_weight=timestamp_cutoff_weight, added_timestamp_degree=added_timestamp_degree)
        for rank in ranks:
            all_timestamp_train_errors[rank] += timestamp_train_errors[rank]
            all_timestamp_test_errors[rank] += timestamp_test_errors[rank]
            all_attempt_train_errors[rank] += attempt_train_errors[rank]
            all_attempt_test_errors[rank] += attempt_test_errors[rank]
            if not is_baseline:
                all_train_accuracies[rank] += train_accuracies[rank]
                all_test_accuracies[rank] += test_accuracies[rank]
        for rank in ranks:
            all_timestamp_train_errors[rank] /= n_splits
            all_timestamp_test_errors[rank] /= n_splits
            all_attempt_train_errors[rank] /= n_splits
            all_attempt_test_errors[rank] /= n_splits
            if not is_baseline:
                all_train_accuracies[rank] /= n_splits
                all_test_accuracies[rank] /= n_splits
    
    if should_print_after:
        print(f"  Timestamp Train Errors: {all_timestamp_train_errors}")
        print(f"  Timestamp Test Errors: {all_timestamp_test_errors}")
        print(f"  Attempt Train Errors: {all_attempt_train_errors}")
        print(f"  Attempt Test Errors: {all_attempt_test_errors}")
        if not is_baseline:
            print(f"  Milestone Attempted Train Accuracy: {all_train_accuracies}")
            print(f"  Milestone Attempted Test Accuracy: {all_test_accuracies}")
    pass



if __name__ == "__main__":

    baseline_tensor = Tensor(filename)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_train_indices = []
    all_test_indices = []
    for train_idx, test_idx in kf.split(baseline_tensor.orig_present_points):
        all_train_indices.append(train_idx)
        all_test_indices.append(test_idx)    


    print("Baseline tensor RMSEs (no modification to identify which milestones are achieved):")
    collect_all_errors(baseline_tensor, ranks, all_train_indices, all_test_indices, n_splits, is_baseline=True)

    print("Add extra achieved milestone slice to feature dimension, RMSEs and accuracy:")
    collect_all_errors(Tensor(filename, contains_whether_achieved=True), ranks, all_train_indices, all_test_indices, n_splits)


    for timestamp_cutoff_weight in timestamp_cutoff_weights:
        for added_timestamp_degree in added_timestamp_degrees:
            print(f"RMSEs and Accuracy using timestamp cutoff weight {timestamp_cutoff_weight}, added timestamp degree {added_timestamp_degree}:")
            collect_all_errors(baseline_tensor, ranks, all_train_indices, all_test_indices, n_splits, timestamp_cutoff_weight=timestamp_cutoff_weight, added_timestamp_degree=added_timestamp_degree)


