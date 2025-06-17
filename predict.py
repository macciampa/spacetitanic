import pandas as pd
import numpy as np
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

# Load the trained pipeline
print("Loading trained pipeline...")
pipeline = joblib.load('titanic_model.joblib')

# Make predictions
print("Making predictions...")
predictions = pipeline.predict(X_test)

# Convert predictions to True/False
predictions = predictions.astype(bool)

# Create submission dataframe
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': predictions
})

# Save predictions to CSV
print("Saving predictions to submission.csv...")
submission_df.to_csv('submission.csv', index=False)
print("Done! Predictions saved to submission.csv") 