import sys
import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from scipy.special import expit
from scipy.optimize import curve_fit


# Helper functions

def power_law(x, a, b):
    return a * np.power(x, b)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def initialize(data, k):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (200, 2)
        k - number of clusters 
    '''
    # initialize the centroids list and add
    # a randomly selected data point to the list
    centroids = []
    centroids.append(data[np.random.randint(
        len(data))])

    # compute remaining k - 1 centroids
    for c_id in range(k - 1):

        # initialize a list to store distances of data
        # points from nearest centroid
        dist = []
        for i in range(len(data)):
            point = data[i]
            d = sys.maxsize

            # compute distance of 'point' from each of the previously
            # selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        # select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist)]
        centroids.append(next_centroid)
        dist = []
    return centroids

def assign_clusters(X, clusters, k):
    for idx in range(len(X)):
        dist = []
        
        curr_x = X[idx]
        
        for i in range(k):
            dis = distance(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

def update_clusters(X, clusters, k):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center
            
            clusters[i]['points'] = []
    return clusters

def pred_cluster(X, clusters, k):
    pred = []
    for i in range(len(X)):
        dist = []
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

def get_max_and_avg_attempts(question_number, file):
    data_csv = pd.read_csv(file)
    data_csv = data_csv[data_csv['Question_Id'] == question_number+1]

    max_attempts = 0
    total_attempts = 0
    total_students_attempted = 0

    for student_number in range(num_learners):
        student_data_csv = data_csv[data_csv['Student_Id'] == student_number+1]
        max_individual_attempts = student_data_csv['Attempt_Count'].max()
        if np.isnan(max_individual_attempts):
            continue

        total_attempts += max_individual_attempts
        if max_individual_attempts > max_attempts:
            max_attempts = max_individual_attempts
        total_students_attempted += 1

    return max_attempts, total_attempts/total_students_attempted




# Load dataset into 3D array

filename = "Getting_Started_Reordered.csv"
data = pd.read_csv(filename)
# print(data.shape)

num_learners = data['Student_Id'].nunique()
num_questions = data['Question_Id'].nunique()
num_attempts = data['Attempt_Count'].nunique()

shaped_data = np.full((num_questions, num_learners, num_attempts), np.nan)

# Fill in with the data points
for row in range(len(data.index) - 2): # Subtract 2 to avoid header and start at 0
    shaped_data[data['Question_Id'][row]-1][data['Student_Id'][row]-1][data['Attempt_Count'][row]-1] = data['Answer_Score'][row]

orig_mask = ~np.isnan(shaped_data)  # True where data is present, False where it is missing
data_tensor = tl.tensor(shaped_data, dtype=tl.float32)
orig_present_points = np.array(np.where(orig_mask)).T




# Use K-fold cross-validation and ALS to factor the tensor for various ranks

ranks = range(6,9)



# Create train tensors
train_tensor = np.copy(data_tensor)

# Optional: assume if student got it right, they get it right every subsequent attempt (rather than empty value)
for question in train_tensor:
    for student in question:
        for attempt_index in range(len(question)):
            if student[attempt_index] == 1:
                student[attempt_index:] = [1 for _ in student[attempt_index:]]
                break


mask = ~np.isnan(train_tensor)
train_tensor = np.nan_to_num(train_tensor)

# Test on different ranks
for rank in ranks:

    weights, factors = parafac(train_tensor, rank=rank, mask=mask, l2_reg=1)
    reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))
    # reconstructed_tensor = expit(reconstructed_tensor) # normalizing to probability


    # Extract prior knowledge (a) and acquired knowledge (b)
    all_extracted_info = []
    all_errors = []

    for question_number, question_matrix in enumerate(reconstructed_tensor):
        # if question_number != 4:
        #     continue

        
        # max_num_attempts, avg_num_attemts = get_max_and_avg_attempts(question_number, filename)
        # print(f"Question {question_number+1}, Max num attempts: {max_num_attempts}, Avg num attempts: {avg_num_attemts}")

        extracted_info_a = []
        extracted_info_b = []

        both_extracted = []

        for student in question_matrix:

            X = np.arange(1, len(student) + 1)

            popt, pcov = curve_fit(power_law, X, student, p0=[1, 1], bounds=([0, 0], [1, 1]))

            extracted_info_a.append(popt[0])
            extracted_info_b.append(popt[1])

            both_extracted.append(list(popt))
        

        
        # call the initialize function to get the centroids
        k = 5
        centroids = initialize(both_extracted, k=k)
        # create clusters from the centroids
        clusters = []
        for i, centroid in enumerate(centroids):
            cluster = {
                'center': centroid,
                'points': []
            }
            clusters.append(cluster)


        clusters = assign_clusters(both_extracted,clusters,k)
        clusters = update_clusters(both_extracted,clusters,k)
        pred = pred_cluster(both_extracted,clusters,k)


        plt.figure()

        plt.scatter(extracted_info_a, extracted_info_b, label='Data', c="blue")
        # plt.title(f'Question {question_number + 1} Learning Curve')
        # plt.title('With K-means ++ Clustering',fontsize=8)
        plt.suptitle(f'Question {question_number + 1} Learning Curve',fontsize=16, y=0.97)
        plt.xlabel("$\t{a}$: prior knowledge")
        plt.ylabel("$\t{b}$: learning rate")

        # x_centroid_vals = [x for x, _ in centroids]
        # y_centroid_vals = [y for _, y in centroids]
        # plt.scatter(x_centroid_vals, y_centroid_vals, color='red')

        # for cluster in clusters:
        #     center = cluster['center']
        #     plt.scatter(center[0],center[1],marker = '^',c = 'red')

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()




