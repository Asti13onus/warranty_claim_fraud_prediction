import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score, mean_squared_error

# Loading the dataset
df = pd.read_csv('df_Clean.csv')

# Checking the shape of the dataset
print(f"Dataset shape: {df.shape}")

# Drop index column
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Checking for null/missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Checking for duplicate values
print(f"Duplicate values: {df.duplicated().sum()}")

# Checking the data types
print(f"Data types:\n{df.dtypes}")

# Unique values in each column
print(f"Unique values:\n{df.nunique()}")

# Renaming the values in product issue columns
issue_mapping = {0: 'No Issue', 1: 'repair', 2: 'replacement'}
for col in ['AC_1001_Issue', 'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue']:
    df[col] = df[col].map(issue_mapping)

# Displaying basic statistics
print(df.describe())

# Displaying the first few rows
print(df.head())

# Plotting distributions
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.subplots_adjust(hspace=0.7)

sns.histplot(x='Region', data=df, ax=ax[0, 0], hue='Fraud', element='bars', fill=True, stat='density', multiple='stack').set(title='Regional Distribution of Fraudulent Claims')
ax[0, 0].xaxis.set_tick_params(rotation=90)

sns.histplot(x='State', data=df, ax=ax[0, 1], hue='Fraud', element='bars', fill=True, stat='density', multiple='stack').set(title='Statewise Distribution of Fraudulent Claims')
ax[0, 1].xaxis.set_tick_params(rotation=90)

sns.histplot(x='City', data=df, ax=ax[1, 0], hue='Fraud', element='bars', fill=True, stat='density', multiple='stack').set(title='Citywise Distribution of Fraudulent Claims')
ax[1, 0].xaxis.set_tick_params(rotation=90)

sns.histplot(x='Area', data=df, ax=ax[1, 1], hue='Fraud', element='bars', fill=True, stat='density', multiple='stack').set(title='Area-wise Distribution of Fraudulent Claims')
ax[1, 1].xaxis.set_tick_params(rotation=90)

plt.show()

# Additional plots
sns.countplot(x='Consumer_profile', data=df, hue='Fraud').set_title('Consumer Profile distribution')
plt.show()

sns.histplot(x='Product_type', data=df, hue='Fraud', multiple='stack').set_title('Product and Fraud Distribution')
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 12))
issue_columns = ['AC_1001_Issue', 'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue']
for i, col in enumerate(issue_columns):
    sns.histplot(x=col, data=df, ax=ax[i // 3, i % 3], hue='Fraud', multiple='stack').set(title=f'{col} and Fraud Distribution')
plt.show()

sns.countplot(x='Service_Centre', data=df, hue='Fraud').set_title('Service Centre and Fraudulent Claims')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(x='Fraud', y='Claim_Value', data=df, ax=ax[0]).set_title('Claim Value and Fraudulent Claims')
sns.violinplot(x='Fraud', y='Claim_Value', data=df, ax=ax[1]).set_title('Claim Value and Fraudulent Claims')
plt.show()

sns.histplot(x='Product_Age', data=df, hue='Fraud', multiple='stack', bins=20).set_title('Product Age (in days) and Fraud Distribution')
plt.show()

sns.histplot(x='Purchased_from', data=df, hue='Fraud', multiple='stack').set_title('Purchased from and Fraudulent Claims')
plt.show()

sns.histplot(x='Call_details', data=df, hue='Fraud', multiple='stack').set_title('Call Duration and Fraudulent Claims')
plt.xlabel('Call Duration (in mins)')
plt.show()

sns.histplot(x='Purpose', data=df, hue='Fraud', multiple='stack').set_title('Purpose and Fraudulent Claims')
plt.show()

# Removing outliers from Claim Value column using IQR method
Q1 = df['Claim_Value'].quantile(0.25)
Q3 = df['Claim_Value'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Claim_Value'] < (Q1 - 1.5 * IQR)) | (df['Claim_Value'] > (Q3 + 1.5 * IQR)))]

# Label encoding categorical columns
le = LabelEncoder()
cols = df.select_dtypes(include=['object']).columns
for col in cols:
    df[col] = le.fit_transform(df[col])
    print(f"{col}: {df[col].unique()}")

# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df.drop('Fraud', axis=1), df['Fraud'], test_size=0.30, random_state=42)

# Decision Tree Classifier
dtree = DecisionTreeClassifier()
param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_leaf': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'criterion': ['gini', 'entropy'],
    'random_state': [0, 42]
}
grid = GridSearchCV(dtree, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best parameters for Decision Tree Classifier: {grid.best_params_}")
dtree = DecisionTreeClassifier(**grid.best_params_)
dtree.fit(X_train, y_train)
print(f"Decision Tree training accuracy: {dtree.score(X_train, y_train)}")
d_pred = dtree.predict(X_test)

# Random Forest Classifier
rfc = RandomForestClassifier()
param_grid = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'criterion': ['gini', 'entropy'],
    'random_state': [0, 42]
}
grid = GridSearchCV(rfc, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best parameters for Random Forest Classifier: {grid.best_params_}")
rfc = RandomForestClassifier(**grid.best_params_)
rfc.fit(X_train, y_train)
print(f"Random Forest training accuracy: {rfc.score(X_train, y_train)}")
r_pred = rfc.predict(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"Logistic Regression training accuracy: {lr.score(X_train, y_train)}")
l_pred = lr.predict(X_test)

# Confusion matrices
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
sns.heatmap(confusion_matrix(y_test, d_pred), annot=True, cmap='coolwarm', ax=ax[0]).set_title('Decision Tree Classifier')
sns.heatmap(confusion_matrix(y_test, r_pred), annot=True, cmap='coolwarm', ax=ax[1]).set_title('Random Forest Classifier')
sns.heatmap(confusion_matrix(y_test, l_pred), annot=True, cmap='coolwarm', ax=ax[2]).set_title('Logistic Regression')
plt.show()

# Classification reports
print("Classification Report for Decision Tree Classifier:")
print(classification_report(y_test, d_pred))

print("Classification Report for Random Forest Classifier:")
print(classification_report(y_test, r_pred))

print("Classification Report for Logistic Regression:")
print(classification_report(y_test, l_pred))

# Model performance metrics
print('==================== Decision Tree Classifier ====================')
print(f'Accuracy Score: {accuracy_score(y_test, d_pred)}')
print(f'R2 Score: {r2_score(y_test, d_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, d_pred)}')

print('==================== Random Forest Classifier ====================')
print(f'Accuracy Score: {accuracy_score(y_test, r_pred)}')
print(f'R2 Score: {r2_score(y_test, r_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, r_pred)}')

print('==================== Logistic Regression ====================')
print(f'Accuracy Score: {accuracy_score(y_test, l_pred)}')
print(f'R2 Score: {r2_score(y_test, l_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, l_pred)}')



