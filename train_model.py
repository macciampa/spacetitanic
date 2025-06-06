import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv('train.csv')

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop(['Transported', 'PassengerId'], axis=1)
y = df['Transported']

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Create a copy of features for preprocessing
X_processed = X.copy()

# Fill missing values
for col in numerical_cols:
    X_processed[col].fillna(X_processed[col].median(), inplace=True)

for col in categorical_cols:
    X_processed[col].fillna(X_processed[col].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    X_processed[col] = le.fit_transform(X_processed[col])

# Scale numerical features
scaler = StandardScaler()
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the trained model
print("\nSaving trained model...")
joblib.dump(model, 'titanic_model.joblib')
print("Model saved as titanic_model.joblib") 