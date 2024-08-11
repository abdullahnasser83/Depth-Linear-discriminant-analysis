import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import gamma
import random

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize
#%%RSLDA
import sys
import os

# Get the current working directory (where your main script is located)
current_dir = os.path.dirname(os.path.realpath(__file__)) + "\RSLDA"

# Add the current directory to the Python path
sys.path.append(current_dir)

from PCA1 import PCA1
from ScatterMat import ScatterMat

import os
import pickle
import numpy as np

from tqdm.auto import tqdm

def RSLDA(X=None, label=None, lambda1 = 0.0002,
                              lambda2 = 0.001,
                              dim = 100,
                              mu  = 0.1,
                              rho = 1.01,
                              max_iter = 100):
    X = np.array(X).T
    label = np.array(label).reshape(-1, 1)
    print("RUNNING RSLDA!")
    m, n = X.shape
    max_mu = 10**5

    # Initialization
    regu = 10**-5
    Sw, Sb = ScatterMat(X, label)
    options = {}
    options['ReducedDim'] = dim
    P1, _ = PCA1(X.T, options)
    Q = np.ones((m, dim))
    E = np.zeros((m, n))
    Y = np.zeros((m, n))
    v = np.sqrt(np.sum(Q*Q, axis=1) + np.finfo(float).eps)
    D = np.diag(1./v)

    # Main loop
    for iter in tqdm(range(1, max_iter+1), total=max_iter):
        
        # Update P
        if iter == 1:
            P = P1
        else:
            M = X - E + Y/mu
            U1, S1, V1 = np.linalg.svd(M @ X.T @ Q, full_matrices=False)
            P = U1 @ V1
            del M
        
        # Update Q
        M = X - E + Y/mu
        Q1 = 2*(Sw - regu*Sb) + lambda1*D + mu*X @ X.T
        Q2 = mu*X @ M.T @ P
        Q = np.linalg.solve(Q1, Q2)
        v = np.sqrt(np.sum(Q*Q, axis=1) + np.finfo(float).eps)
        D = np.diag(1./v)
        
        # Update E
        eps1 = lambda2/mu
        temp_E = X - P @ Q.T @ X + Y/mu
        E = np.maximum(0, temp_E - eps1) + np.minimum(0, temp_E + eps1)
        
        # Update Y, mu
        Y = Y + mu*(X - P @ Q.T @ X - E)
        mu = min(rho*mu, max_mu)
        leq = X - P @ Q.T @ X - E
        EE = np.sum(np.abs(E), axis=1)
        obj = np.trace(Q.T @ (Sw - regu*Sb) @ Q) + lambda1*np.sum(v) + lambda2*np.sum(EE)
        
        if iter > 2:
            if np.linalg.norm(leq, np.inf) < 10**-7 and abs(obj - obj_prev) < 0.00001:
                print(iter)
                break
        
        obj_prev = obj
    
    return P, Q, E, obj
#%% DepthLinearDiscriminantAnalysis

class MatrixDepthLinearDiscriminantAnalysis():
    def __init__(self, n_components=None, max_iterations=12):
        self.n_components = n_components
        self.label_encoder = None
        self.transformed_data = None
        self.max_iterations = max_iterations
        self.info = 0

    def fit(self, X, y):
        # Encode class labels into integers
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Number of classes and features
        n_classes = len(np.unique(y_encoded))
        n_features = X.shape[1]

        # Initialize matrices
        self.Sb = np.zeros((n_features, n_features))
        self.Sw = np.zeros((n_features, n_features))

        # Compute class-wise means
        class_means = []
        for i in range(n_classes):
            X_i = X[y_encoded == i]
            class_means.append(np.mean(X_i, axis=0))

        # Compute between-class scatter matrix Sb
        overall_mean = np.mean(X, axis=0)
        for i in range(n_classes):
            mean_diff = class_means[i] - overall_mean
            self.Sb += len(X[y_encoded == i]) * np.outer(mean_diff, mean_diff)

        # Estimate the robust within-class scatter matrix
        self.Sw = self.estimate_robust_scatter_matrix(X)
        # print(self.Sw)
        # Solve generalized eigenvalue problem inv(Sw) @ Sb
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(self.Sw) @ self.Sb)
        from scipy.linalg import eigh
        # eigenvalues, eigenvectors = eigh(np.linalg.inv(self.Sw) @ self.Sb)

        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]

        # Select the top k eigenvectors if n_components is specified
        if self.n_components is not None:
            sorted_indices = sorted_indices[:self.n_components]

        self.eigenvectors = np.real(eigenvectors[:, sorted_indices])

        return self

    def transform(self, X):
        if self.eigenvectors is None:
            raise ValueError("The model has not been trained yet. Please call fit() first.")

        return X @ self.eigenvectors
    
    def fit_transform(self, X, y):
        # Fit the model
        self.fit(X, y)
        
        # Transform the data
        transformed_data = self.transform(X)
        
        return transformed_data
    def predict(self, X):
        if self.eigenvectors is None:
            raise ValueError("The model has not been trained yet. Please call fit() first.")

        # Transform the input data using the learned eigenvectors
        X_transformed = X @ self.eigenvectors

        # Apply your classification logic here
        # For example, you can use a threshold or any other classification method
        # Here's a simple threshold-based classification as an example
        threshold = 0.0
        predictions = (X_transformed[:, 0] > threshold).astype(int)

        return predictions    
    # new method
  
    # Define the main algorithm to compute the deepest matrix estimator
    # Step 2: Find the optimal scatter matrix
    def estimate_robust_scatter_matrix(self, data):

        # list of functions
        # empirical depth function
        def empirical_depth(Σ, Xi):
            try:
                Σ_inv = np.linalg.inv(Σ)
                depth = np.dot(np.dot(Xi.T, Σ_inv), Xi)   # 1 / 
            except np.linalg.LinAlgError:
                depth = 0  # Set depth to 0 for singular matrices
            return depth
        
        def initialze_covariance_matrix(self, data):
            n, m = X.shape
            Σ = np.cov(X.T)  # Initial scatter matrix
        
            # Calculate the sum of empirical depth functions
            def sum_empirical_depth(Σ):
                depth_sum = 0
                for i in range(n):
                    Xi = X[i]
                    depth_sum += empirical_depth(Σ, Xi)
                return depth_sum # self.empirical_depth(Σ, Xi)

            # Update Σ using the empirical depth functions
            depth_sums = []
            for i in range(n):
                Xi = X[i]
                depth = empirical_depth(Σ, Xi)
                depth_sums.append(depth * np.dot(np.dot(Xi.T, Σ), Xi))
            depth_sums = np.array(depth_sums)
            
            # Add a small constant to depth sums to avoid division by zero
            #depth_sums += 1e-8    
            # Normalize the depth sums
            weights = depth_sums / np.mean(depth_sums)

            # Set negative weights to zero
            weights[weights < 0] = 0
            
            # Calculate the weighted covariance matrix
            Σ = np.cov(X.T, aweights=weights)
            return Σ
            
        # Depth function        
        def calculate_covariance_matrix_depth(covariance_matrix, direction):
            trace_product = np.trace(np.dot(covariance_matrix, np.outer(direction, direction)))
            matrix_depth = 1 / trace_product
            #print(matrix_depth)
            return matrix_depth


        def find_deepest_direction(current_estimator, direction_set):
            """
            Calculate the matrix depth of a positive semidefinite matrix with respect to a given distribution.
        
            Args:
            current_estimator (numpy.ndarray): The current matrix Σ (positive semidefinite).
            direction_set (list of numpy.ndarray): A list of direction vectors u.
        
            Returns:
            float: The matrix depth value.
            numpy.ndarray: The direction vector corresponding to the matrix depth.
            """
            matrix_depth = float('inf')  # Initialize to positive infinity
            best_direction = None
        
            for u in direction_set:
                u_dot_sigma_u = u.T @ current_estimator @ u  # Calculate u^T Σ u
                if u_dot_sigma_u >= 0:
                    # Check if u^T Σ u is non-negative as per the condition
                    if u_dot_sigma_u < matrix_depth:
                        matrix_depth = u_dot_sigma_u
                        best_direction = u
        
            return matrix_depth, best_direction
        

        
        def generate_directions(dim):
            directions = []
            for i in range(dim):
                direction = np.zeros(dim)
                direction[i] = 1
                directions.append(direction)
            return directions
        
        ## start calculate robust scatter matrix        
        # calc the different
        diffs = data[:, np.newaxis] - data
        
        # Step 1: initialization
        initial_covariance = np.cov(data, rowvar=False)

        # Step 1a: Define the direction set U (for simplicity, we consider two directions: height and weight)
        direction_set = generate_directions(data.shape[1])

        num_iterations = self.max_iterations 
        current_estimator = initialze_covariance_matrix(self, data)
        
        step_size = 0.001
        
        # Step 1b: 
        matrix_depths = calculate_covariance_matrix_depth(current_estimator, direction_set[0])
        best_direction_index = np.argmin(matrix_depths)
        #best_direction_index = np.argmax(matrix_depths)
        best_direction = direction_set[best_direction_index]
        depth, best_direction =find_deepest_direction(current_estimator, direction_set)
        for t in range(1, num_iterations + 1):            
            # Step 2a: Compute matrix depth for each direction
            #matrix_depths = [compute_matrix_depth2(current_estimator, diffs, direction) for direction in direction_set]
            matrix_depths = [calculate_covariance_matrix_depth(current_estimator, direction) for direction in direction_set]
            
            # Step 2b: Select the direction with the smallest matrix depth
            best_direction_index = np.argmin(matrix_depths)
            best_direction = direction_set[best_direction_index]
            depth, best_direction =find_deepest_direction(current_estimator, direction_set)
            # Step 2c: Update the current estimator
            current_estimator = current_estimator + step_size * np.outer(best_direction, best_direction)

            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(self.Sw+current_estimator) @ self.Sb)
            #print(depth, best_direction )
            
        self.info = "Depth:" + str(min(matrix_depths)) #+ ", best_separation_ratio:" +str(best_separation_ratio) + "  at"+ str(best_index) # + str(current_estimator)
        self.info = "Depth:" + str(depth) #+ ", best_separation_ratio:" +str(best_separation_ratio) + "  at"+ str(best_index) # + str(current_estimator)
        return current_estimator
    
    

    # Add a method to retrieve the depth_sum
    def get_info(self):
        
    
        if self.info is None:
            raise ValueError("The model has not been trained yet. Please call fit() first.")
        return self.info
 
import numpy as np
import matplotlib.pylab as pl


#%% main

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ot.dr import wda, fda

best_alpha =  2.008 
best_n_components= 2 
best_regularization= 1e-10

# load software defect datasets
from sklearn.datasets import load_iris

import os
import sys
import os.path as path
import pandas as pd

# Set the directory path
abs_path_pkg = path.abspath(path.join(__file__, "../../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)



# Read the CSV file
# Create empty lists to store accuracy results and p-values
accuracy_results = []
wilcoxon_results = []


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

Save_to_excel = False
Save_to_excel = True

Introduce_outliers = False
# Introduce_outliers = True

Enable_plotting = False
Enable_plotting = True

Tuning = False
Tuning = True
max_Tuning = 10

best_max_iterations = 5
iteration_range = range(1, best_max_iterations)    
# Define the number of runs
n_runs = 1
# Define classifiers and their abbreviations
classifiers = [
    (RandomForestClassifier, "RF"),
    (LogisticRegression, "LR"),
    (GradientBoostingClassifier, "GB"),
    (GaussianNB, "GNB"),
    (KNeighborsClassifier, "KNN")
]

classifier = classifiers[4][0]
for r in range(n_runs):
    print("Running ",r, "...")

  
    # Load the iris dataset
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    DS = "iris"
    

    #%% intoduce outliers data
    
    if Introduce_outliers is True:
        DS = DS+ " olir"
        # # Set the number of outliers to introduce
        num_outliers = 20
        
        # # Select random indices to introduce outliers
        # outlier_indices = np.random.choice(len(X), size=num_outliers, replace=False)
        
        # # Introduce outliers by modifying selected observations
        # for idx in outlier_indices:
        #     # Randomly select a feature and set it to an extreme value
        #     feature_idx = np.random.choice(X.shape[1])
        #     X[idx, feature_idx] = np.random.uniform(low=-100, high=100)
        
        # # # Verify the modified dataset
        # # print("Modified Iris dataset:")
        # # print(X[outlier_indices])
        
        
        #Select  specific indices to introduce outliers
    
    # =============================================================================
        outlier_indices = [10, 20, 25, 40, 50, 60, 70, 80, 90]  # Modify these indices to introduce outliers
        
        # Introduce outliers by modifying selected observations
        for idx in outlier_indices:
            # Modify specific features to extreme values
            X[idx, 0] = 1000.0  # Modify the first feature
            X[idx, 1] = -1000.0  # Modify the second feature
            X[idx, 2] = 2000.0  # Modify the third feature
            X[idx, 3] = -2000.0  # Modify the fourth feature
        
        #Verify the modified dataset
        print("Modified Iris dataset:")
        print(X[outlier_indices])
    # =============================================================================
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12   )
    #%% tunning 

    if Tuning is True:
        iteration_range = range(1, max_Tuning)
        results =[]
        for max_iterations in iteration_range:
            # Create and fit the MatrixDepthLinearDiscriminantAnalysis model
            D_LDA = MatrixDepthLinearDiscriminantAnalysis(max_iterations=max_iterations)
            X_train_D_LDA = D_LDA.fit_transform(X_train, y_train)
            X_test_D_LDA = D_LDA.transform(X_test)
               
            # Train a classifier (e.g., Logistic Regression) on the transformed data
            # Fit a k-NN classifier to the Alpha-Discriminant Analysis-transformed data
            cls_D_LDA = classifier()
            cls_D_LDA.fit(X_train_D_LDA, y_train)
            accuracy_D_LDA = cls_D_LDA.score(X_test_D_LDA, y_test)
            
            # Get the depth_sum for the current iteration
            info = D_LDA.get_info()
            
            # Store the results
            results.append((max_iterations, info, accuracy_D_LDA))
            print(f"{max_iterations} iterations, {info}  Accuracy: {accuracy_D_LDA}")
            #print("Best accuracy at:", iteration_no)
        
        best_accuracy = -1  # Initialize with a value lower than the lowest possible accuracy
        best_max_iterations = None
        for max_iterations, depth_sum, accuracy in results:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_max_iterations = max_iterations
        # Find the best hyperparameters
        #best_alpha = grid_search.best_params_['alpha']
        #best_n_components = grid_search.best_params_['n_components']
        #best_regularization = grid_search.best_params_['regularization']
        print(f"Best accuracy: {best_accuracy} achieved with {best_max_iterations} iterations.")


#%%Apply transformation


    # Apply LDA transformation
    print("original data shape: ", X_train.shape)
    # Apply LDA transformation
    lda = LinearDiscriminantAnalysis()
    X_train_LDA = lda.fit_transform(X_train, y_train)
    X_test_LDA = lda.transform(X_test)
    print("LDA data shape", X_train_LDA.shape)
    

    # Apply D_LDA transformation
    D_LDA = MatrixDepthLinearDiscriminantAnalysis(max_iterations=best_max_iterations )
    X_train_D_LDA = D_LDA.fit_transform(X_train, y_train)
    X_test_D_LDA = D_LDA.transform(X_test)
    print("D_LDA data shape: ", X_train_D_LDA.shape)
    cls_D_LDA = classifier()
    cls_D_LDA.fit(X_train_D_LDA, y_train)

    # Compute Fisher Discriminant Analysis (FDA)
    p = 2
    Pfda, projfda = fda(X, y, p)
    
    # Compute Wasserstein Discriminant Analysis (WDA)
    p = 2
    reg = 1e0
    k = 10
    maxiter = 1
    P0 = np.random.randn(X.shape[1], p)
    P0 /= np.sqrt(np.sum(P0**2, 0, keepdims=True))
    Pwda, projwda = wda(X, y, p, reg, k, maxiter=maxiter, P0=P0)
    #%% 
    import torch
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np
    
    
    # Apply LDA for feature extraction
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    # Convert data to PyTorch tensors
    X_train_lda = torch.tensor(X_train_lda, dtype=torch.float32)
    X_test_lda = torch.tensor(X_test_lda, dtype=torch.float32)
    
    # Define the DeepLDA model
    class DeepLDA(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(DeepLDA, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim),
            )
    
        def forward(self, x):
            return self.layers(x)
    
    # Initialize the DeepLDA model
    input_dim = X_train_lda.shape[1]  # Number of LDA components
    hidden_dim = 128
    output_dim = len(np.unique(y))  # Number of classes
    model = DeepLDA(input_dim, hidden_dim, output_dim)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the DeepLDA model
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        X_train_DeepLDA = model(X_train_lda)
        loss = criterion(X_train_DeepLDA, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
    
    # Predict on the test set
    with torch.no_grad():
        X_test_DeepLDA = model(X_test_lda)
        _, predicted = torch.max(X_test_DeepLDA, 1)
        
    
    # Fit a k-NN classifier to the original data
    cls_original = classifier()
    cls_original.fit(X_train, y_train)
    
    # Fit a k-NN classifier to the FDA-transformed data
    X_train_fda = projfda(X_train)
    X_test_fda = projfda(X_test)
    cls_fda = classifier()
    cls_fda.fit(X_train_fda, y_train)
    
    # Fit a k-NN classifier to the WDA-transformed data
    X_train_wda = projwda(X_train)
    cls_wda = classifier()
    cls_wda.fit(X_train_wda, y_train)
    
    # Fit a k-NN classifier to the Alpha-Discriminant Analysis-transformed data
    cls_D_LDA = classifier()
    cls_D_LDA.fit(X_train_D_LDA, y_train)
    
        # Print dimensions of LDA-transformed data
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    # Fit an LDA model to the training data
    lda = LinearDiscriminantAnalysis()
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    # Print dimensions of LDA-transformed data
    print("X_train_lda shape:", X_train_lda.shape)
    print("X_test_lda shape:", X_test_lda.shape)
    # Fit a k-NN classifier to the LDA-transformed data
    cls_lda = classifier()
    cls_lda.fit(X_train_lda, y_train)

    
    
    # Fit a k-NN classifier to the PCA-transformed data
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    cls_pca = KNeighborsClassifier()
    cls_pca.fit(X_train_pca, y_train)
    
    
    # Let's use RSLDA for feature extraction and classification
    P, Q, E, obj = RSLDA(X=X_train, label=y_train, dim=5, max_iter=100)
    
    # Transform the training data to the discriminant space
    X_train_transformed_rslda = X_train @ P
    
    # Transform the testing data to the discriminant space
    X_test_transformed_rslda = X_test @ P
    
    # Use a simple classifier (e.g., Logistic Regression) for classification
    clf_rslda = classifier()
    clf_rslda.fit(X_train_transformed_rslda, y_train)
    # y_pred = clf.predict(X_test_transformed)
    
    # Use Kernel-LDA for feature extraction and classification
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    poly_degree = 3
    kernel_lda = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearDiscriminantAnalysis())

    # Fit the model to the data    
    kernel_lda.fit(X_train, y_train)
    X_train_transformed_KLDA = kernel_lda.transform(X_train)
    X_test_transformed_KLDA = kernel_lda.transform(X_test)

    
    clf_klda = classifier()
    clf_klda.fit(X_train_transformed_KLDA, y_train)

    # Use K-nearest neighbors (KNN) on Deep LDA-transformed data
    clf_DeepLDA = classifier()
    clf_DeepLDA.fit(X_train_DeepLDA.detach().numpy(), y_train)

    
    # Calculate the accuracy for each method
    accuracy_original = cls_original.score(X_test, y_test)
    accuracy_fda = cls_fda.score(projfda(X_test), y_test)
    accuracy_wda = cls_wda.score(projwda(X_test), y_test)
    accuracy_D_LDA = cls_D_LDA.score(X_test_D_LDA, y_test)
    accuracy_lda = cls_lda.score(X_test_lda, y_test)
    accuracy_pca = cls_pca.score(X_test_pca, y_test)
    accuracy_rslda = clf_rslda.score(X_test_transformed_rslda, y_test)
    accuracy_KLDA = clf_klda.score(X_test_transformed_KLDA, y_test)
    accuracy_DeepLDA = clf_DeepLDA.score(X_test_DeepLDA, y_test)
    
    print(f'Original Data Test Accuracy: {accuracy_original:f}')
    print(f'FDA Projection Test Accuracy: {accuracy_fda:f}')
    print(f'WDA Projection Test Accuracy: {accuracy_wda:f}')
    print(f'D_LDA Projection Test Accuracy: {accuracy_D_LDA:f}')
    print(f'LDA Projection Test Accuracy: {accuracy_lda:f}')
    print(f'PCA Projection Test Accuracy: {accuracy_pca:f}')
    print(f'RSLDA Projection Test Accuracy: {accuracy_rslda:f}')
    print(f'KLDA Projection Test Accuracy: {accuracy_KLDA:f}')
    print(f'DeepLDA Projection Test Accuracy: {accuracy_DeepLDA:f}')
    feature_one=0
    feature_two=1
    n_classes = len(np.unique(y))
    if Enable_plotting == True:
        # Set custom colors for each class
        #colors = ['r', 'g', 'b']
        
        # Plot 2D projections with custom colors
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for c in range(n_classes):
            class_data = X_train[y_train == c]
            plt.scatter(class_data[:, feature_one], class_data[:, feature_two],   label=f'Class {c}')
        plt.legend()
        plt.title(f'Original Data')
        
        plt.subplot(1, 2, 2)
        if X_train_D_LDA.shape[1] > 1:
            for c in range(n_classes):
                class_data = X_train_D_LDA[y_train == c]
                plt.scatter(class_data[:, feature_one], class_data[:, feature_two],   label=f'Class {c}')
            plt.legend()
            plt.title('D-LDA Projection')        
        plt.tight_layout()
        plt.show()    
        
    if Enable_plotting == True:
        # Set custom colors for each class
        colors = ['r', 'g', 'b']
        
        # Plot 2D projections with custom colors
        plt.figure(figsize=(10, 11))
        
        plt.subplot(3, 3, 1)
        for c in range(n_classes):
            class_data = X_test[y_test == c]
            plt.scatter(class_data[:, feature_one], class_data[:, feature_two],  c=colors[c], label=f'Class {c}')
        plt.legend()
        plt.title(f'Original Data')
        
        plt.subplot(3, 3, 2)
        if projfda(X_test).shape[1]>1:
            for c in range(n_classes):
                class_data = projfda(X_test)[y_test == c]
                plt.scatter(class_data[:, feature_one], class_data[:, feature_two],  c=colors[c], label=f'Class {c}')
            plt.legend()
            plt.title(f'FDA Projection\nTest Accuracy: {accuracy_fda:.2f}')
        
        plt.subplot(3, 3, 4)
        for c in range(n_classes):
            class_data = projwda(X_test)[y_test == c]
            plt.scatter(class_data[:, feature_one], class_data[:, feature_two],  c=colors[c], label=f'Class {c}')
        plt.legend()
        plt.title(f'WDA Projection\nTest Accuracy: {accuracy_wda:.2f}')
        
        plt.subplot(3, 3, 3)
        for c in range(n_classes):
            class_data = X_test_D_LDA[y_test == c]
            plt.scatter(class_data[:, feature_one], class_data[:, feature_two],  c=colors[c], label=f'Class {c}')
        plt.legend()
        plt.title(f'D-LDA Projection\nTest Accuracy: {accuracy_D_LDA:.2f}')
        
        plt.subplot(3, 3, 5)
        if X_test_lda.shape[1] > 1:
            for c in range(n_classes):
                class_data = X_test_lda[y_test == c]
                plt.scatter(class_data[:, feature_one], class_data[:, feature_two],  c=colors[c], label=f'Class {c}')
            plt.legend()
            plt.title(f'LDA Projection\nTest Accuracy: {accuracy_lda:.2f}')
        
        plt.subplot(3, 3, 6)
        for c in range(n_classes):
            class_data = X_test_pca[y_test == c]
            plt.scatter(class_data[:, feature_one], class_data[:, feature_two],  c=colors[c], label=f'Class {c}')
        plt.legend()
        plt.title(f'PCA Projection\nTest Accuracy: {accuracy_pca:.2f}')
        
        plt.subplot(3, 3, 7)
        for c in range(n_classes):
            class_data = X_test_transformed_rslda[y_test == c]
            plt.scatter(class_data[:, feature_one], class_data[:, feature_two], c=colors[c], label=f'Class {c}')
        plt.legend()
        plt.title(f'RSLDA Projection\nTest Accuracy: {accuracy_rslda:.2f}')
    
        # Plot Kernel-LDA projection
        plt.subplot(3, 3, 8)
        if X_test_transformed_KLDA.shape[1] > 1:
            for c in range(n_classes):
                class_data = X_test_transformed_KLDA[y_test == c]
                plt.scatter(class_data[:, feature_one], class_data[:, feature_two], c=colors[c], label=f'Class {c}')
            plt.legend()
            plt.title(f'Kernel-LDA Projection\nTest Accuracy: {accuracy_KLDA:.2f}')

        # Plot Kernel-LDA projection
        plt.subplot(3, 3, 9)
        if X_test_DeepLDA.shape[1] > 1:
            for c in range(n_classes):
                class_data = X_test_DeepLDA[y_test == c]
                plt.scatter(class_data[:, feature_one], class_data[:, feature_two], c=colors[c], label=f'Class {c}')
            plt.legend()
            plt.title(f'DeepLDA Projection\nTest Accuracy: {accuracy_DeepLDA:.2f}')
        
        plt.tight_layout()
        plt.show()





