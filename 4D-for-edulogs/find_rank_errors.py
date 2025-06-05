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


    counter = 0
    for train_indices, test_indices in kf.split(example_tensor.orig_present_points):

        # Keep track of progress
        print(f"Iteration {counter + 1} out of {n_splits}")
        counter += 1

        # Create train tensor
        train_tensor = np.copy(initial_tensor.data_tensor)
        
        # Fill in train tensor with NaNs where the test values are
        for test_index in test_indices:
            tensor_test_index = initial_tensor.orig_present_points[test_index]
            train_tensor[tensor_test_index[0]][tensor_test_index[1]][tensor_test_index[2]] = np.nan

        mask = ~np.isnan(train_tensor)
        train_tensor = np.nan_to_num(train_tensor)

        # Test on different ranks
        for rank in ranks:

            weights, factors = parafac(train_tensor, rank=rank, mask=mask, l2_reg=l2)
            reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

            if is_stratified:

                reconstructed_tensor = stratify_points(reconstructed_tensor)
                
                # Compute accuracy (only applicable when running stratifying_points, as the accuracy looks at whether the points are exactly equal)
                train_acc, test_acc = find_accuracy(initial_tensor.data_tensor, reconstructed_tensor, test_indices, initial_tensor.orig_present_points)
                train_accuracy[rank].append(train_acc)
                test_accuracy[rank].append(test_acc)

            total = 0
            num = 0
            for outside in reconstructed_tensor:
                for middle in outside:
                    for attempt in middle:
                        if not np.isnan(attempt):
                            total += attempt
                            num += 1
            average_values[rank].append(total/num)

            # Compute the errors
            mse_train_values, mse_test_values = [], []

            for test_index in test_indices:
                tensor_test_index = initial_tensor.orig_present_points[test_index]
                mse_test_values.append((initial_tensor.data_tensor[tensor_test_index[0]][tensor_test_index[1]][tensor_test_index[2]] - reconstructed_tensor[tensor_test_index[0]][tensor_test_index[1]][tensor_test_index[2]]) ** 2)

            for train_index in train_indices:
                tensor_test_index = initial_tensor.orig_present_points[train_index]
                mse_train_values.append((initial_tensor.data_tensor[tensor_test_index[0]][tensor_test_index[1]][tensor_test_index[2]] - reconstructed_tensor[tensor_test_index[0]][tensor_test_index[1]][tensor_test_index[2]]) ** 2)

            train_mse = np.mean(mse_train_values)
            test_mse = np.mean(mse_test_values)

            train_errors[rank].append(train_mse)
            test_errors[rank].append(test_mse)

    if is_stratified:
        average_train_accuracy = {rank: np.mean(accuracy) for rank, accuracy in train_accuracy.items()}
        average_test_accuracy = {rank: np.mean(accuracy) for rank, accuracy in test_accuracy.items()}
        print("Train accuracy: ", average_train_accuracy)
        print("Test accuracy: ", average_test_accuracy)

    average_train_error = {rank: np.mean(errors) for rank, errors in train_errors.items()}
    print("Train errors: ", average_train_error)

    average_test_error = {rank: np.mean(errors) for rank, errors in test_errors.items()}
    print("Test errors: ", average_test_error)

    average_average_values = {rank: np.mean(avg) for rank, avg in average_values.items()}
    print(f"Average tensor value: {average_average_values}")
