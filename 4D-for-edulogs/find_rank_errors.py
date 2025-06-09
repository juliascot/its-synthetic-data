'''
This file decomposes and reconstructs the tensor for various ranks, printing the root mean squared error
and accuracy.

Set desired filename, is_stratified, ranks, l2 (within parafac call), and n_splits (within KFold)
'''

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensor import Tensor

filename = "parsing/file_wrangler_third_pass_results.csv"
ranks = range(1, 9)
l2 = 0 # Regularization -- basically to what degree we ignore potential outliers.
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


def find_completed_milestones(tensor: np.ndarray) -> np.ndarray:
    completed_milestone_slice = np.full((len(tensor), len(tensor[0])), 0)
    for s in range(len(tensor)):
        for m in range(len(tensor[0])):
            if not np.isnan(tensor[s][m][0]):
                completed_milestone_slice[s][m] = 1
    return completed_milestone_slice



def decomp_and_errors(orig_tensor_class: Tensor,
                      ranks: list[int],
                      is_baseline: bool = False, 
                      timestamp_cutoff_weight: float = None, 
                      added_timestamp_degree: float = None,
                      should_print_after: bool = True
    ) -> tuple[dict[int: float], dict[int: float], dict[int: float]]:

    ranks = [rank for rank in ranks if rank <= min(orig_tensor_class.data_tensor.shape)]

    timestamp_errors = {rank: None for rank in ranks}
    attempt_errors = {rank: None for rank in ranks}
    accuracies = {rank: None for rank in ranks}
    avg_filler_timestamps = {rank: None for rank in ranks}

    tensor_copy = np.copy(orig_tensor_class.data_tensor)
    orig_milestones_completed = find_completed_milestones(tensor_copy)

    if timestamp_cutoff_weight is not None:
        tensor_copy = add_extreme_timestamps(tensor_copy, added_timestamp_degree * orig_tensor_class.max_time)

    mask = ~np.isnan(tensor_copy)
    tensor_copy = np.nan_to_num(tensor_copy)

    for rank in ranks:

        weights, factors = parafac(tensor_copy, rank=rank, mask=mask, l2_reg=l2)
        reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

        if not is_baseline:
            if timestamp_cutoff_weight is not None:
                completed_milestone_slice = generate_completed_milestone_values_slice(reconstructed_tensor, orig_tensor_class.max_time)
                accuracy = find_accuracy(orig_milestones_completed, completed_milestone_slice)
            else:
                reconstructed_tensor = stratify_points(reconstructed_tensor)
                accuracy = find_accuracy(orig_tensor_class.data_tensor[:, :, 2], reconstructed_tensor[:, :, 2])

            accuracies[rank] = accuracy

        timestamp_errors[rank] = np.mean((orig_tensor_class.data_tensor[:, :, 0] - reconstructed_tensor[:, :, 0])**2)**0.5
        attempt_errors[rank] = np.mean((orig_tensor_class.data_tensor[:, :, 1] - reconstructed_tensor[:, :, 1])**2)**0.5

        absent_timestamps = (orig_milestones_completed == 0)
        avg_filler_timestamps[rank] = np.mean(reconstructed_tensor[:, :, 0][absent_timestamps])

    if should_print_after:
        print(f"  Timestamp Errors: {timestamp_errors}")
        print(f"  Attempt Errors: {attempt_errors}")
        if not is_baseline:
            print(f"  Milestone Attempted Accuracy: {accuracies}")
        print(f"  Average Filler Timestamps: {avg_filler_timestamps}")

    return timestamp_errors, attempt_errors, accuracies



if __name__ == "__main__":

    baseline_tensor = Tensor(filename)
    print(f"Maximum original timestamp: {baseline_tensor.max_time}")

    print("Baseline tensor RMSEs (no modification to identify which milestones are achieved):")
    decomp_and_errors(baseline_tensor, ranks, is_baseline=True)

    for timestamp_cutoff_weight in timestamp_cutoff_weights:
        for added_timestamp_degree in added_timestamp_degrees:
            print(f"RMSEs and Accuracy using timestamp cutoff weight {timestamp_cutoff_weight}, added timestamp degree {added_timestamp_degree}:")
            decomp_and_errors(baseline_tensor, ranks, timestamp_cutoff_weight=timestamp_cutoff_weight, added_timestamp_degree=added_timestamp_degree)


