#!/usr/bin/env python
# coding: utf-8

# # Importing the packages and libraries that are required for the project

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, brier_score_loss
)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import StandardScaler


# # Loading Data

# In[2]:


# Default dataset path
default_path = "heart.csv"

# Ask user for dataset path or use default
file_path = input(f"Enter the path to the dataset (default: {default_path}): ") or default_path

# Load the dataset
data = pd.read_csv(file_path)

data = pd.read_csv("heart.csv")
data.head()


# In[3]:


data.info()


# # Evaluate Target Distribution

# In[4]:


#first of all let us evaluate the target and find out if our data is imbalanced or not
cols= ["#6daa9f","#774571"]
sns.countplot(x= data["DEATH_EVENT"], palette= cols)


# # Correlation Matrix Analysis

# In[5]:


#Examaning a corelation matrix of all the features 
cmap = sns.diverging_palette(275,150,  s=40, l=65, n=9)
corrmat = data.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(corrmat,cmap= cmap,annot=True, square=True);


# # Age Distribution Visualization

# In[6]:


#Evauating age distribution 
plt.figure(figsize=(20,12))
#colours =["#774571","#b398af","#f1f1f1" ,"#afcdc7", "#6daa9f"]
Days_of_week=sns.countplot(x=data['age'],data=data, hue ="DEATH_EVENT",palette = cols)
Days_of_week.set_title("Distribution Of Age", color="#774571")


# # Boxen and Swarm Plot Analysis

# In[7]:


# Boxen and swarm plot of some non binary features.
feature = ["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium", "time"]
for i in feature:
    plt.figure(figsize=(8,8))
    sns.swarmplot(x=data["DEATH_EVENT"], y=data[i], color="black", alpha=0.5)
    sns.boxenplot(x=data["DEATH_EVENT"], y=data[i], palette=cols)
    plt.show()


# In[8]:


sns.kdeplot(x=data["time"], y=data["age"], hue =data["DEATH_EVENT"], palette=cols)


# In[9]:


data.describe().T


# # Feature and Target Assignment

# In[10]:


#assigning values to features as X and target as y
X=data.drop(["DEATH_EVENT"],axis=1)
y=data["DEATH_EVENT"]


# # Feature Scaling Setup

# In[11]:


#Set up a standard scaler for the features
# Set up a standard scaler for the features
col_names = list(X.columns)
s_scaler = StandardScaler()  # Directly use StandardScaler
X_df = s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names)
X_df.describe().T


# # Train-Test Split and Standardization

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# # Manual Confusion Matrix Metrics Calculation

# In[13]:


# Function to calculate metrics from confusion matrix
def calc_metrics(confusion_matrix):
    TP, FN = confusion_matrix[0][0], confusion_matrix[0][1]
    FP, TN = confusion_matrix[1][0], confusion_matrix[1][1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (TN + FP)
    FNR = FN / (TP + FN)
    Precision = TP / (TP + FP)
    F1_measure = 2 * TP / (2 * TP + FP + FN)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error_rate = (FP + FN) / (TP + FP + FN + TN)
    BACC = (TPR + TNR) / 2
    TSS = TPR - FPR
    HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    metrics = [TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS]
    return metrics


# # Model Metrics Calculation Function

# In[14]:


# Function to fit model and calculate metrics
def get_metrics(model, X_train, X_test, y_train, y_test, LSTM_flag):
    metrics = []
    if LSTM_flag == 1:
        # Convert data to numpy array
        Xtrain, Xtest, ytrain, ytest = map(np.array, [X_train, X_test, y_train, y_test])
        # Reshape data
        shape = Xtrain.shape
        Xtrain_reshaped = Xtrain.reshape(len(Xtrain), shape[1], 1)
        Xtest_reshaped = Xtest.reshape(len(Xtest), shape[1], 1)
        model.fit(Xtrain_reshaped, ytrain, epochs=50, validation_data=(Xtest_reshaped, ytest), verbose=0)
        lstm_scores = model.evaluate(Xtest_reshaped, ytest, verbose=0)
        predict_prob = model.predict(Xtest_reshaped)
        pred_labels = predict_prob > 0.5
        pred_labels_1 = pred_labels.astype(int)
        matrix = confusion_matrix(ytest, pred_labels_1, labels=[1, 0])
        lstm_brier_score = brier_score_loss(ytest, predict_prob)
        lstm_roc_auc = roc_auc_score(ytest, predict_prob)
        metrics.extend(calc_metrics(matrix))
        metrics.extend([lstm_brier_score, lstm_roc_auc, lstm_scores[1]])
    elif LSTM_flag == 0:
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(y_test, predicted, labels=[1, 0])
        model_brier_score = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
        model_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        metrics.extend(calc_metrics(matrix))
        metrics.extend([model_brier_score, model_roc_auc, model.score(X_test, y_test)])
    return metrics


# # Selecting Classification Algorithms
# 
# ## Chosen Classification Algorithms
#    ###  * K-Nearest Neighbor (KNN)
#    ###  * Random Forest (RF)
#    ###  * Support Vector Machine (SVM)
# 
# ## Selected Deep Learning Algorithm
#    ### * Long Short-Term Memory (LSTM)

# # Parameter Tuning for KNN

# In[15]:


# Define KNN parameters for grid search
knn_parameters = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}

# Create KNN model
knn_model = KNeighborsClassifier()

# Perform grid search with cross-validation
knn_cv = GridSearchCV(knn_model, knn_parameters, cv=10, n_jobs=-1)

# Fit the model with training data
knn_cv.fit(X_train_std, y_train)

# Print the best parameters found by GridSearchCV
print("\nBest Parameters for KNN based on GridSearchCV: ", knn_cv.best_params_)
print()


# In[16]:


best_n_neighbors = knn_cv.best_params_['n_neighbors']


#  # Parameter Tuning for RandomForest

# In[17]:


# Define Random Forest parameters for grid search
param_grid_rf = {
    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "min_samples_split": [2, 4, 6, 8, 10],
}

# Create Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=10, n_jobs=-1, scoring="accuracy")

# Fit the model with training data
grid_search_rf.fit(X_train_std, y_train)

# Display the best parameters from the grid search
best_rf_params = grid_search_rf.best_params_
print("\nBest Parameters for Random Forest based on GridSearchCV: ", best_rf_params)
print()

# Extract and print the best values for 'min_samples_split' and 'n_estimators'
best_min_samples_split = best_rf_params['min_samples_split']
best_n_estimators = best_rf_params['n_estimators']
print(f"Best 'min_samples_split': {best_min_samples_split}")
print(f"Best 'n_estimators': {best_n_estimators}")


# #  Parameter Tuning for SVM

# In[18]:


# Define Support Vector Machine parameters for grid search
param_grid_svc = {"kernel": ["linear"], "C": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

# Create Support Vector Machine model
svc_classifier = SVC()

# Perform grid search with cross-validation
grid_search_svc = GridSearchCV(estimator=svc_classifier, param_grid=param_grid_svc, cv=10, n_jobs=-1, scoring="accuracy")

# Fit the model with training data
grid_search_svc.fit(X_train_std, y_train)

# Display the best parameters from the grid search
best_svc_params = grid_search_svc.best_params_
print("\nBest Parameters for Support Vector Machine based on GridSearchCV: ", best_svc_params)
print()

# Extract and print the best value for 'C'
C_value = best_svc_params['C']
print(f"Best 'C' value: {C_value}")


# # 10-Fold Cross-Validation with Performance Metrics for KNN, Random Forest, SVM, and LSTM Models

# In[19]:


# Define Stratified K-Fold cross-validator
cv_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

# Initialize metric columns
metric_columns = [
    'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Precision',
    'F1_measure', 'Accuracy', 'Error_rate', 'BACC', 'TSS', 'HSS',
    'Brier_score', 'AUC', 'Acc_by_package_fn'
]

# Initialize metrics lists for each algorithm
knn_metrics_list, rf_metrics_list, svm_metrics_list, lstm_metrics_list = [], [], [], []

# Placeholder for best parameters
best_n_neighbors = 3
min_samples_split = 4
n_estimators = 50
C = 1.0

# 10 Iterations of 10-fold cross-validation
for iter_num, (train_index, test_index) in enumerate(cv_stratified.split(X_train_std, y_train), start=1):
    # KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)

    # Random Forest Model
    rf_model = RandomForestClassifier(min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=42)

    # SVM Classifier Model
    svm_model = SVC(C=C, kernel='linear', probability=True, random_state=42)

    # LSTM Model
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(X_train_std.shape[1], 1)))
    lstm_model.add(LSTM(64, activation='relu', return_sequences=False))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Split data into training and testing sets
    features_train, features_test = X_train_std[train_index], X_train_std[test_index]
    labels_train, labels_test = y_train.iloc[train_index], y_train.iloc[test_index]

    # Reshape data for LSTM
    features_train_reshaped = features_train.reshape(features_train.shape[0], features_train.shape[1], 1)
    features_test_reshaped = features_test.reshape(features_test.shape[0], features_test.shape[1], 1)

    # Get metrics for each algorithm
    knn_metrics = get_metrics(knn_model, features_train, features_test, labels_train, labels_test, LSTM_flag=0)
    rf_metrics = get_metrics(rf_model, features_train, features_test, labels_train, labels_test, LSTM_flag=0)
    svm_metrics = get_metrics(svm_model, features_train, features_test, labels_train, labels_test, LSTM_flag=0)
    lstm_metrics = get_metrics(lstm_model, features_train_reshaped, features_test_reshaped, labels_train, labels_test, LSTM_flag=1)

    # Append metrics to respective lists
    knn_metrics_list.append(knn_metrics)
    rf_metrics_list.append(rf_metrics)
    svm_metrics_list.append(svm_metrics)
    lstm_metrics_list.append(lstm_metrics)

    # Create a DataFrame for all metrics
    metrics_all_df = pd.DataFrame(
        [knn_metrics, rf_metrics, svm_metrics, lstm_metrics],
        columns=metric_columns,
        index=['KNN', 'RF', 'SVM', 'LSTM']
    )

    # Display metrics for all algorithms in each iteration
    print(f'\nIteration {iter_num}: \n')
    print(f'\n----- Metrics for all Algorithms in Iteration {iter_num} -----\n')
    print(metrics_all_df.round(decimals=2).T)
    print('\n')


# # Generate and Display Metrics DataFrames for KNN, Random Forest, SVM, and LSTM Across Iterations

# In[20]:


# Initialize metric index for each iteration
metric_index_df = ['iter1', 'iter2', 'iter3', 'iter4', 'iter5', 'iter6', 'iter7', 'iter8', 'iter9', 'iter10']

# Create DataFrames for each algorithm's metrics
knn_metrics_df = pd.DataFrame(knn_metrics_list, columns=metric_columns, index=metric_index_df)
rf_metrics_df = pd.DataFrame(rf_metrics_list, columns=metric_columns, index=metric_index_df)
svm_metrics_df = pd.DataFrame(svm_metrics_list, columns=metric_columns, index=metric_index_df)
lstm_metrics_df = pd.DataFrame(lstm_metrics_list, columns=metric_columns, index=metric_index_df)

# Display metrics for each algorithm in each iteration
for i, (name, metrics_df) in enumerate(
    zip(["KNN", "Random Forest", "SVM", "LSTM"], [knn_metrics_df, rf_metrics_df, svm_metrics_df, lstm_metrics_df]), start=1
):
    print(f'\nMetrics for Algorithm {name}:\n')
    print(metrics_df.round(decimals=2).T)
    print('\n')


# # Calculate the average metrics for each algorithm

# In[21]:


knn_avg_df = knn_metrics_df.mean()
rf_avg_df = rf_metrics_df.mean()
svm_avg_df = svm_metrics_df.mean()
lstm_avg_df = lstm_metrics_df.mean()

# Create a DataFrame with the average performance for each algorithm
avg_performance_df = pd.DataFrame({
    'KNN': knn_avg_df, 
    'RF': rf_avg_df, 
    'SVM': svm_avg_df, 
    'LSTM': lstm_avg_df
}, index=metric_columns)

# Display the average performance for each algorithm
print("\nAverage Performance Metrics for Each Algorithm:\n")
print(avg_performance_df.round(decimals=2))


# # Evaluating the performance of various algorithms by comparing their ROC curves and AUC scores on the test dataset

# In[22]:


# Evaluate KNN
knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn_model.fit(X_train_std, y_train)
y_score_knn = knn_model.predict_proba(X_test_std)[:, 1]

# Compute ROC curve and ROC area
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_score_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label='KNN ROC curve (area = {:.2f})'.format(roc_auc_knn))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[23]:


# Evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=42)
rf_model.fit(X_train_std, y_train)
y_score_rf = rf_model.predict_proba(X_test_std)[:, 1]

# Compute ROC curve and ROC area
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest ROC curve (area = {:.2f})'.format(roc_auc_rf))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[24]:


# Evaluate SVM
svm_model = SVC(C=C, kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_std, y_train)
y_score_svm = svm_model.predict_proba(X_test_std)[:, 1]

# Compute ROC curve and ROC area
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='SVM ROC curve (area = {:.2f})'.format(roc_auc_svm))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[25]:


# Evaluate LSTM

# Reshape test data for LSTM
features_test_reshaped = X_test_std.reshape(X_test_std.shape[0], X_test_std.shape[1], 1)

# Predicted probabilities
y_score_lstm = lstm_model.predict(features_test_reshaped)

# Compute ROC curve and ROC area
fpr_lstm, tpr_lstm, _ = roc_curve(y_test, y_score_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr_lstm, tpr_lstm, color='purple', lw=2, label='LSTM ROC curve (area = {:.2f})'.format(roc_auc_lstm))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LSTM ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[26]:


# Print the average performance metrics for each algorithm
print("\nAverage Performance Metrics for Each Algorithm:\n")
print(avg_performance_df.round(decimals=2))
print('\n')


# In[27]:


# Plot bar chart for average metrics
avg_performance_df.T.plot(kind='bar', figsize=(12, 8), width=0.8)

# Add labels and title
plt.title('Average Performance Metrics for Each Algorithm', fontsize=16)
plt.ylabel('Metric Values', fontsize=14)
plt.xlabel('Algorithms', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


# In this project, we tested different algorithms to predict heart disease outcomes using patient data. The models we used were:
# 
# K-Nearest Neighbors (KNN)
# Random Forest (RF)
# Support Vector Machine (SVM)
# Long Short-Term Memory (LSTM)
# 
# Best-Performing Algorithm: Random Forest
# Random Forest gave the best results compared to the other models. It performed well in terms of accuracy, precision, and recall. Here’s why:
# 
# Handles Complex Data:
# Random Forest can handle complex relationships in the data better than many other algorithms.
# 
# Avoids Overfitting:
# By using multiple decision trees, Random Forest makes predictions that are more balanced and reliable.
# 
# Shows Important Features:
# It also highlights which features (like age or blood pressure) are most important for predicting outcomes.
# 
# Why Not Other Algorithms?
# 
# KNN: It didn’t perform as well because it’s sensitive to outliers and high-dimensional data.
# SVM: While it gave good results, it required more time and tuning to work properly.
# LSTM: This deep learning model needs more data to perform its best. It works better for time-series data than the kind used here.
# 
# Random Forest was the best choice for this project because it gave accurate results and worked well with the data we had.
