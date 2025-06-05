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
                      ):
    pass



if __name__ == "__main__":

    # Put the file into a tensor
    initial_tensor = Tensor(filename)

    # Use K-fold cross-validation and ALS to factor the tensor for various ranks
    train_errors, test_errors, train_accuracy, test_accuracy = {rank: [] for rank in ranks}, {rank: [] for rank in ranks}, {rank: [] for rank in ranks}, {rank: [] for rank in ranks}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) # If the data is too sparse, high ranks will throw errors, 
    # but we can sometimes get around it by using high n_splits

    average_values = {rank: [] for rank in ranks}

    counter = 0
    for train_indices, test_indices in kf.split(initial_tensor.orig_present_points):

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
