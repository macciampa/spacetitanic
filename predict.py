import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the test data
print("Loading test data...")
test_df = pd.read_csv('test.csv')

# Save PassengerId for submission
passenger_ids = test_df['PassengerId']

# Prepare features (same as in training)
X_test = test_df.drop(['PassengerId'], axis=1)

# Handle categorical variables
categorical_cols = X_test.select_dtypes(include=['object']).columns
numerical_cols = X_test.select_dtypes(include=['int64', 'float64']).columns

# Create a copy of features for preprocessing
X_test_processed = X_test.copy()

# Fill missing values
for col in numerical_cols:
    X_test_processed[col].fillna(X_test_processed[col].median(), inplace=True)

for col in categorical_cols:
    X_test_processed[col].fillna(X_test_processed[col].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    X_test_processed[col] = le.fit_transform(X_test_processed[col])

# Scale numerical features
scaler = StandardScaler()
X_test_processed[numerical_cols] = scaler.fit_transform(X_test_processed[numerical_cols])

# Load the trained model
print("Loading trained model...")
model = joblib.load('titanic_model.joblib')

# Make predictions
print("Making predictions...")
predictions = model.predict(X_test_processed)

# Create submission dataframe
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': predictions
})

# Save predictions to CSV
print("Saving predictions to submission.csv...")
submission_df.to_csv('submission.csv', index=False)
print("Done! Predictions saved to submission.csv") 