import numpy as np
import pandas as pd
from catboost import Pool
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import the data and normalize them
data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data = data.dropna()
X = data.drop(columns=['target'])
y = data['target']
print(data['target'].value_counts(normalize=True))
print(len(data_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_out = data_test


# Check the distribution after oversampling
print(y_train.value_counts(normalize=True))

# Define the hyperparameters and their possible values to search
param_grid = {
    'iterations': [200, 500],
    'learning_rate': [0.1] 
}

# Create a CatBoost Classifier
catboost = CatBoostClassifier()

# Use Stratified K-Folds Cross-Validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Grid Search with StratifiedKFold cross-validation
grid_search = GridSearchCV(estimator=catboost, param_grid=param_grid, cv=stratified_kfold)
grid_search.fit(X_train, y_train)

# Print the best parameters and their corresponding score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Predict on the test set
pred = grid_search.predict(X_test)
list_of_pred = pred.tolist()

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, list_of_pred)
conf_matrix = confusion_matrix(y_test, pred)

# Extract TP, TN, FP, FN from the confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()

print("Out of sample accuracy:", accuracy)
print("accuracyTn:", tn / (tn + fn))
print("accuracyTp:", tp / (tp + fp))

# Predict on the test data
test_pred = grid_search.predict(X_out)
list_testpred = test_pred.tolist()

# Save predictions to a file
with open("predictions.txt", "w") as fh:
    for i in list_testpred:
        fh.write(str(i))
