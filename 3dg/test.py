import numpy as np
import tensorly as tl
import pandas as pd
from tensorly.decomposition import parafac
from sklearn.model_selection import KFold

# Helper functions

def stratify_points(tensor): # Makes points 0 or 1 -- only for 3-way tensors
    for student in range(len(tensor)):
        for question in range(len(tensor[0])):
            for attempt in range(len(tensor[0][0])):
                if tensor[student][question][attempt] >= 0.5:
                    tensor[student][question][attempt] = 1
                else:
                    tensor[student][question][attempt] = 0
    return tensor

def find_accuracy(orig_tensor, constructed_tensor, test_indices, orig_present_points): # Reports train and test accuracy

    correct_test = 0
    correct_train = 0
    num_test_points = len(test_indices)
    num_train_points = len(orig_present_points) - num_test_points

    for index in range(num_train_points):
        tensor_index = orig_present_points[index]
        if index in test_indices:
            if orig_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]] == constructed_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]]:
                correct_test += 1
        else:
            if orig_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]] == constructed_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]]:
                correct_train += 1
    
    return correct_train / num_train_points, correct_test / num_test_points




# Load dataset into 3D array

filename = ""
data = pd.read_csv(filename)
# print(data.shape)

num_learners = data['Student_Id'].nunique()
num_questions = data['Question_Id'].nunique()
num_attempts = data['Attempt_Count'].nunique()

shaped_data = np.full((num_learners, num_questions, num_attempts), np.nan)

# Fill in with the data points
for row in range(len(data.index) - 2): # Subtract 2 to avoid header and start at 0
    shaped_data[data['Student_Id'][row]-1][data['Question_Id'][row]-1][data['Attempt_Count'][row]-1] = data['Answer_Score'][row]

orig_mask = ~np.isnan(shaped_data)  # True where data is present, False where it is missing
data_tensor = tl.tensor(shaped_data, dtype=tl.float32)
orig_present_points = np.array(np.where(orig_mask)).T




# Use K-fold cross-validation and ALS to factor the tensor for various ranks

ranks = range(1,5)
train_errors, test_errors, train_accuracy, test_accuracy = {rank: [] for rank in ranks}, {rank: [] for rank in ranks}, {rank: [] for rank in ranks}, {rank: [] for rank in ranks}
kf = KFold(n_splits=30, shuffle=True, random_state=42) # If the data is too sparse, high ranks will throw errors, but you can sometimes get around it by using high n_splits
stratify = True # Set this to true if you want results to have the data round to zeros or ones (will also print the accuracy)

for train_indices, test_indices in kf.split(orig_present_points):

    # Create train tensors
    train_tensor = np.copy(data_tensor)

    # Optional: assume if student got it right, they get it right every subsequent attempt (rather than empty value)
    for student in train_tensor:
        for question in student:
            for attempt_index in range(len(question)):
                if question[attempt_index] == 1:
                    question[attempt_index:] = [1 for _ in question[attempt_index:]]
                    break
    
    # Fill in train tensor with NaNs where the test values are
    for test_index in test_indices:
        tensor_index = orig_present_points[test_index]
        train_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]] = np.nan

    mask = ~np.isnan(train_tensor)
    train_tensor = np.nan_to_num(train_tensor)

    # Test on different ranks
    for rank in ranks:

        weights, factors = parafac(train_tensor, rank=rank, mask=mask)
        reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))


        if stratify:

            reconstructed_tensor = stratify_points(reconstructed_tensor) # This line is optional -- depends on whether you want decimal approximations or reformat to just ones and zeros
            
            # Compute accuracy (only applicable when running stratifying_points, as the accuracy looks at whether the points are exactly equal)
            train_acc, test_acc = find_accuracy(data_tensor, reconstructed_tensor, test_indices, orig_present_points)
            train_accuracy[rank].append(train_acc)
            test_accuracy[rank].append(test_acc)

        # Compute the errors
        mse_train_values, mse_test_values = [], []

        for test_index in test_indices:
            tensor_index = orig_present_points[test_index]
            mse_test_values.append((data_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]] - reconstructed_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]]) ** 2)

        for train_index in train_indices:
            tensor_index = orig_present_points[train_index]
            mse_train_values.append((data_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]] - reconstructed_tensor[tensor_index[0]][tensor_index[1]][tensor_index[2]]) ** 2)

        train_mse = np.mean(mse_train_values)
        test_mse = np.mean(mse_test_values)

        train_errors[rank].append(train_mse)
        test_errors[rank].append(test_mse)


if stratify:
    average_train_accuracy = {rank: np.mean(accuracy) for rank, accuracy in train_accuracy.items()}
    average_test_accuracy = {rank: np.mean(accuracy) for rank, accuracy in test_accuracy.items()}
    print("Train accuracy: ", average_train_accuracy)
    print("Test accuracy: ", average_test_accuracy)

average_train_error = {rank: np.mean(errors) for rank, errors in train_errors.items()}
average_test_error = {rank: np.mean(errors) for rank, errors in test_errors.items()}
print("Train errors: ", average_train_error)
print("Test errors: ", average_test_error)
