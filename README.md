# **Diabetes Prediction Using Machine Learning**

## Introduction
Diabetes is a chronic disease affecting millions worldwide. This project aims to develop and compare multiple machine learning models to predict diabetes using a dataset containing various medical predictor variables. 

## Dataset Overview
The dataset used in this project is `diabetes.csv`, which includes multiple features such as glucose levels, BMI, age, and blood pressure, among others. The target variable is `Outcome`, indicating whether a person has diabetes (1) or not (0).

## Code Implementation and Explanation
### Importing Libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
```
**Explanation:** This section imports necessary Python libraries for data manipulation, visualization, machine learning models, and deep learning.

### Loading and Exploring Data
```python
df = pd.read_csv("diabetes.csv")
df.head()
df.isnull().sum()
df.duplicated().sum()
df.shape
df.info()
df.describe()
```
**Explanation:** This section loads the dataset and checks for missing values, duplicate records, dataset shape, and descriptive statistics.

### Data Visualization
```python
df.hist(figsize=(12,8))
plt.suptitle("Histogram")
df.boxplot(figsize=(12,10))
plt.title("Boxplot")
plt.xticks(rotation = 45)
plt.show()

sns.pairplot(df, hue="Outcome", palette='coolwarm', diag_kind='kde')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```
**Explanation:**
- **Histograms:** Displayed the distribution of each feature.
- **Boxplots:** Help identify outliers in the dataset.
- **Pairplot:** Visualizes feature relationships.
- **Heatmap:** Displays correlations between variables.
- **Countplot & Pie Chart:** Illustrated the distribution of diabetic vs non-diabetic patients.

### Data Preprocessing
```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```
**Explanation:**
- Splits data into independent (`X`) and dependent (`y`) variables.
- Scales the features to normalize values.
- Splits data into training (80%) and testing (20%) sets.

### Model Training and Evaluation
Each model is trained and evaluated based on accuracy and classification reports.

## Machine Learning Models Implemented
A variety of models were trained and evaluated:

1. **Logistic Regression (LR)**
   - Used for binary classification.
   - Provides probability estimates for class predictions.
   - Strengths: Simple and interpretable.
   - Weaknesses: Assumes linearity between independent variables and the log-odds.

2. **K-Nearest Neighbors (KNN)**
   - Classification based on nearest neighbors.
   - Strengths: Non-parametric, simple.
   - Weaknesses: Sensitive to outliers and high-dimensional data.

3. **Decision Tree (DT)**
   - Recursive partitioning to split data.
   - Strengths: Easy to interpret.
   - Weaknesses: Prone to overfitting.

4. **Random Forest (RF)**
   - Ensemble of decision trees.
   - Strengths: Handles overfitting well.
   - Weaknesses: Computationally expensive.

5. **AdaBoost (ABC)**
   - Boosting method that improves weak classifiers.
   - Strengths: Reduces bias and variance.
   - Weaknesses: Sensitive to noise.

6. **Gradient Boosting (GBC)**
   - Iteratively improves weak learners.
   - Strengths: High accuracy.
   - Weaknesses: Computationally intensive.

7. **Support Vector Machine (SVM)**
   - Finds an optimal hyperplane for classification.
   - Strengths: Effective in high-dimensional spaces.
   - Weaknesses: Slow on large datasets.

8. **Naïve Bayes (NB)**
   - Based on Bayes' theorem.
   - Strengths: Works well with small datasets.
   - Weaknesses: Assumes feature independence.

9. **XGBoost (XGB)**
   - Gradient boosting framework optimized for performance.
   - Strengths: Handles missing values well.
   - Weaknesses: Requires hyperparameter tuning.

10. **LightGBM (LGB)**
    - Faster gradient boosting.
    - Strengths: High efficiency.
    - Weaknesses: May not perform well on small datasets.

#### Logistic Regression
```python
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.4f}%')
print(classification_report(y_test, y_pred))
```
**Why Logistic Regression?** It is a simple yet effective baseline model for binary classification.

#### K-Nearest Neighbors (KNN)
```python
model1 = KNeighborsClassifier()
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy1*100:.4f}%')
print(classification_report(y_test, y_pred))
```
**Why KNN?** It classifies based on the majority class of the nearest neighbors, making it useful for pattern recognition.

#### Decision Tree, Random Forest, AdaBoost, Gradient Boosting, SVM, Naïve Bayes, XGBoost, LightGBM
```python
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "GaussianNB": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(),
    "LightGBM": lgb.LGBMClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy*100:.4f}%')
    print(classification_report(y_test, y_pred))
```
**Why these models?** They offer different strengths in handling data complexity, overfitting, and performance trade-offs.

### Model Accuracy Comparison
```python
MODEL = [accuracy*100 for accuracy in [accuracy, accuracy1]]
Name = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'AdaBoost', 'GradientBoosting', 'SVM', 'GaussianNB', 'XGBoost', 'LightGBM']
plt.figure(figsize=(10,5))
sns.barplot(x=Name,y=MODEL)
plt.xlabel('Models')
plt.ylabel('Accuracy %')
plt.title('Model Accuracy Comparison')
plt.show()
```
**Why Compare Models?** To determine which model performs best for predicting diabetes.

### Deep Learning Model Implementation

A **Neural Network** was implemented using TensorFlow/Keras:
- **Input Layer**: 8 neurons (one for each feature)
- **Hidden Layers**: 2 layers with 8 neurons each (ReLU activation)
- **Output Layer**: 1 neuron (Sigmoid activation for binary classification)
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Epochs**: 150
- **Batch Size**: 42


```python
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=42)
loss, acc = model.evaluate(X_test, y_test)
print(f'Neural Network Accuracy: {acc*100:.2f}%')
```
**Why Neural Networks?** They capture complex relationships in data and improve prediction accuracy.

### Feature Importance Visualization
```python
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
plt.figure(figsize=(8,6))
sns.barplot(x=X.columns, y=importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance using Random Forest')
plt.xticks(rotation=45)
plt.show()
```
**Why Feature Importance?** To identify the most influential features affecting diabetes predictions.

## Conclusion
- **Ensemble models** like linearRegression, GradientBoosting, and GaussianNB performed best.
- **Feature importance** analysis highlighted Glucose, BMI, and Age as crucial features.
- **Neural networks** demonstrated promising accuracy for diabetes prediction.
