# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Dataset .csv')

# Display basic info
print(data.info())
print(data.describe())

# Drop irrelevant columns
data = data.drop(['Restaurant ID', 'Restaurant Name', 'Address', 'Locality',  'Locality Verbose', 'Longitude', 'Latitude', 'Rating color', 
                  'Rating text', 'Switch to order menu', 'Currency'], axis=1)

# Check for missing values
print("\nMissing values before filling:\n", data.isnull().sum())

# Fill missing values
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])
    else:
        data[column] = data[column].fillna(data[column].median())

print("\nMissing values after filling:\n", data.isnull().sum())

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
X = data.drop('Aggregate rating', axis=1)
y = data['Aggregate rating']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 1. Linear Regression Model
# ===============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluation
print("\n=== Linear Regression Evaluation ===")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))

# ===============================
# 2. Decision Tree Regression Model
# ===============================
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluation
print("\n=== Decision Tree Regression Evaluation ===")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_dt))
print("R² Score:", r2_score(y_test, y_pred_dt))

# ===============================
# Feature Importance (Decision Tree)
# ===============================
feature_importances = pd.DataFrame({'Feature': X.columns,'Importance': dt_model.feature_importances_}).sort_values(by='Importance', ascending=False)

print("\n=== Feature Importances (Decision Tree) ===")
print(feature_importances)

# Plot Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()
