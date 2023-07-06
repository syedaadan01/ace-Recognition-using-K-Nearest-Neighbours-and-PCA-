# face-Recognition-using-K-Nearest-Neighbours-and-PCA-
#approach :The Face Recognition approach using K Nearest Neighbors (KNN) and Principal Component Analysis (PCA) involves two key steps: dimensionality reduction using PCA and classification using KNN.

Dimensionality Reduction with PCA: PCA is used to reduce the dimensionality of the face images while preserving the most important information. It achieves this by finding the principal components that capture the maximum variance in the dataset. These principal components represent a lower-dimensional representation of the face images, allowing for efficient computation and improved accuracy in classification.

Classification with KNN: After reducing the dimensionality using PCA, KNN is employed for classification. KNN is a simple yet effective algorithm that classifies data based on their similarity to other data points. In the context of face recognition, KNN compares the reduced-dimensional representation of a test face image with the representations of known face images in the training set. It assigns a label to the test image based on the majority vote of its K nearest neighbors in the training set.

By combining PCA for dimensionality reduction and KNN for classification, the Face Recognition system can accurately identify individuals by comparing their facial features with the stored representations in the training dataset. This approach is widely used due to its simplicity and effectiveness in face recognition tasks.
# requirement : 1
#Pre-process the dataset by normalizing each face image vector to unit length (i.e., dividing 
each vector by its magnitude). Next, for each of the 10 subjects, randomly select 150 images 
for training and use the remaining 20 for testing. Create such random splits a total of 5 times. 
You must carry out each of the following experiments 5 times and report average accuracy 
and standard deviation over the 5 random splits, as well as computation times.
# code :
# TASK :1 Requirement 1 

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('fea.csv')  # Replace 'path/to' with the actual path to the dataset file

normalized_df = df.div(np.linalg.norm(df, axis=1), axis=0)

train_data = pd.DataFrame()  # Empty DataFrame for training data
test_data = pd.DataFrame()  # Empty DataFrame for testing data

for subject in range(10):
    subject_data = normalized_df.iloc[subject * 170: (subject + 1) * 170]  # Select data for each subject
    shuffled_data = subject_data.sample(frac=1)  # Shuffle the data for random selection

    train_data = pd.concat([train_data, shuffled_data[:150]], ignore_index=True)  # Select the first 150 instances for training
    test_data = pd.concat([test_data, shuffled_data[150:]], ignore_index=True)  # Select the remaining 20 instances for testing

X = pd.concat([train_data.iloc[:, 1:], test_data.iloc[:, 1:]], ignore_index=True)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(pd.concat([train_data.iloc[:, 0], test_data.iloc[:, 0]], ignore_index=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train = X_scaled[:len(train_data)]
y_train = y[:len(train_data)]

X_test = X_scaled[len(train_data):]
y_test = y[len(train_data):]

pca = PCA(n_components=0.95)  # Choose the desired amount of variance to explain, such as 95%
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')  # Choose the desired hyperparameters
knn.fit(X_train_pca, y_train)

y_pred = knn.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
# req 2 :
#TASK 1 : Requiremnent 2 

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def mahalanobis_distance(x1, x2, covariance_matrix):
    return cdist(x1.reshape(1, -1), x2.reshape(1, -1), 'mahalanobis', VI=covariance_matrix)

def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return dot_product / (norm_x1 * norm_x2)

def k_nearest_neighbors(x_test, x_train, y_train, distance_measure, k, covariance_matrix=None):
    distances = []
    for i in range(len(x_train)):
        if distance_measure == 'euclidean':
            distance = euclidean_distance(x_test, x_train[i])
        elif distance_measure == 'mahalanobis':
            distance = mahalanobis_distance(x_test, x_train[i], covariance_matrix)
        elif distance_measure == 'cosine':
            distance = cosine_similarity(x_test, x_train[i])
        distances.append((distance, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    classes = [neighbor[1] for neighbor in neighbors]
    return max(set(classes), key=classes.count)

def evaluate_accuracy(x_test, x_train, y_train, distance_measure, k, covariance_matrix=None):
    predicted_labels = np.array([k_nearest_neighbors(x_test[i], x_train, y_train, distance_measure, k, covariance_matrix) for i in range(len(x_test))])
    correct_predictions = np.sum(predicted_labels == y_test)
    accuracy = correct_predictions / len(x_test)
    return accuracy

df = pd.read_csv('fea.csv')  # Replace 'path/to' with the actual path to the dataset file

normalized_df = df.div(np.linalg.norm(df, axis=1), axis=0)

train_data = pd.DataFrame()  # Empty DataFrame for training data
test_data = pd.DataFrame()  # Empty DataFrame for testing data

for subject in range(10):
    subject_data = normalized_df.iloc[subject * 170: (subject + 1) * 170]  # Select data for each subject
    shuffled_data = subject_data.sample(frac=1)  # Shuffle the data for random selection

    train_data = pd.concat([train_data, shuffled_data[:100]], ignore_index=True)  # Select the first 100 instances for training
    test_data = pd.concat([test_data, shuffled_data[100:170]], ignore_index=True)  # Select the remaining 70 instances for testing

x_train = train_data.iloc[:, 1:].to_numpy()
y_train = train_data.iloc[:, 0].to_numpy()

x_test = test_data.iloc[:, 1:].to_numpy()
y_test = test_data.iloc[:, 0].to_numpy()

covariance_matrix = np.cov(x_train.T)  # Compute covariance matrix once

distance_measures = ['euclidean', 'mahalanobis', 'cosine']
k_values = [1, 3, 5]

for distance_measure in distance_measures:
    for k in k_values:
        accuracy = evaluate_accuracy(x_test, x_train, y_train, distance_measure, k, covariance_matrix)
        print(f"Distance Measure: {distance_measure}, k: {k}, Accuracy: {accuracy:.2%}")
        #req 3 : 
        import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('fea.csv')  # Replace 'path/to' with the actual path to the dataset file

normalized_df = df.div(np.linalg.norm(df, axis=1), axis=0)

train_data = pd.DataFrame()  # Empty DataFrame for training data
test_data = pd.DataFrame()  # Empty DataFrame for testing data

label_encoder = LabelEncoder()  # Instantiate the LabelEncoder object

for subject in range(10):
    subject_data = normalized_df.iloc[subject * 170: (subject + 1) * 170]  # Select data for each subject
    shuffled_data = subject_data.sample(frac=1)  # Shuffle the data for random selection

    train_data = pd.concat([train_data, shuffled_data[:100]], ignore_index=True)  # Select the first 100 instances for training
    test_data = pd.concat([test_data, shuffled_data[100:170]], ignore_index=True)  # Select the remaining 70 instances for testing

X_train = train_data.iloc[:, 1:].to_numpy()
y_train = train_data.iloc[:, 0].to_numpy()

X_test = test_data.iloc[:, 1:].to_numpy()
y_test = test_data.iloc[:, 0].to_numpy()

# Fit and transform the label encoder on the training labels
y_train_encoded = label_encoder.fit_transform(y_train)

# Check for unseen labels in the test set
unseen_labels = np.setdiff1d(y_test, label_encoder.classes_)
if unseen_labels.size > 0:
    print(f"Test set contains previously unseen labels: {unseen_labels}")
    # Filter out the unseen labels from the test set
    mask = np.isin(y_test, label_encoder.classes_)
    X_test = X_test[mask]
    y_test = y_test[mask]

best_accuracy = 0
best_num_components = 0

for num_components in range(1, X_train.shape[1] + 1):
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train)

    # Check if the test set is empty after filtering unseen labels
    if X_test.shape[0] > 0:
        X_test_pca = pca.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=5)  # Choose the desired number of neighbors
        knn.fit(X_train_pca, y_train_encoded)

        y_pred = knn.predict(X_test_pca)
        accuracy = accuracy_score(y_test_encoded, y_pred)

        print(f"Number of Components: {num_components}, Accuracy: {accuracy:.2%}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_components = num_components
    else:
        print(f"Number of Components: {num_components}, No test samples")

print(f"Best Number of Components: {best_num_components}, Best Accuracy: {best_accuracy:.2%}")
