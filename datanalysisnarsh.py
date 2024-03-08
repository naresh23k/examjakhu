import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('C:\\Users\\Admin\\Downloads\\Data (1).csv')

# Feature Engineering and Data Preparation
# For example, if 'ed' is a categorical variable, encode it
# encoder = OneHotEncoder(sparse=False)
# encoded_ed = encoder.fit_transform(data[['ed']])
# data['ed_encoded'] = encoded_ed

# Splitting data into features and target variable
X = data.drop('custcat', axis=1)  # Features
y = data['custcat']  # Target variable

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with preprocessing and KNN classifier
preprocessor = ColumnTransformer(
    transformers=[
        # Add transformations for categorical variables if any
        # ('encode', OneHotEncoder(), ['ed']),
        ('scale', StandardScaler(), X.columns)
    ]
)

knn_pipeline = make_pipeline(preprocessor, KNeighborsClassifier())

# Hyperparameter Tuning
# You can perform a grid search over different values of n_neighbors
# and other hyperparameters using GridSearchCV

# Fit the model
knn_pipeline.fit(X_train, y_train)

# Predictions
y_pred = knn_pipeline.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
# Evaluate the model using cross-validation
cv_scores = cross_val_score(knn_pipeline, X, y, cv=5)
print("Cross-Validation Mean Accuracy:", np.mean(cv_scores))

# Visualization
# Add more visualizations here to gain insights into the data and model
