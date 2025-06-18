import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
os.makedirs('data_out', exist_ok=True)

def train_model():
    # Load the data
    print("Loading training data...")
    df = pd.read_csv('data_in/train.csv')

    # Create Infant feature
    df['Infant'] = df['Age'].fillna(-1) <= 4

    # Create TotalSpent feature
    spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpent'] = df[spending_columns].fillna(0).sum(axis=1)

    # Split Cabin into Deck and Side features
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Side'] = df['Side'].fillna('Unknown')  # Handle missing values
    df['Deck'] = df['Deck'].fillna('Unknown')  # Handle missing values

    # Display basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

    # Separate features and target
    X = df.drop(['Transported', 'PassengerId', 'Name', 'Cabin', 'Num'], axis=1)
    y = df['Transported']

    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

    # Create full pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])

    # Train model on all data
    print("\nTraining model...")
    pipeline.fit(X, y)

    # Get feature names after preprocessing
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Save feature names to text file
    with open('data_out/feature_names.txt', 'w') as f:
        f.write("All feature names after preprocessing:\n")
        f.write("=" * 50 + "\n")
        for i, feature_name in enumerate(feature_names, 1):
            f.write(f"{i:3d}. {feature_name}\n")
        f.write(f"\nTotal number of features after preprocessing: {len(feature_names)}")
    print(f"\nFeature names saved to data_out/feature_names.txt")

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Save the trained pipeline
    print("\nSaving trained pipeline...")
    joblib.dump(pipeline, 'data_out/titanic_model.joblib')
    print("Pipeline saved as data_out/titanic_model.joblib")
    
    return pipeline

def make_predictions(pipeline=None):
    # Load the test data
    print("Loading test data...")
    test_df = pd.read_csv('data_in/test.csv')

    # Create Infant feature
    test_df['Infant'] = test_df['Age'].fillna(-1) <= 4

    # Create TotalSpent feature
    spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    test_df['TotalSpent'] = test_df[spending_columns].fillna(0).sum(axis=1)

    # Split Cabin into Deck and Side features
    test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)
    test_df['Side'] = test_df['Side'].fillna('Unknown')  # Handle missing values
    test_df['Deck'] = test_df['Deck'].fillna('Unknown')  # Handle missing values

    # Save PassengerId for submission
    passenger_ids = test_df['PassengerId']

    # Prepare features (same as in training)
    X_test = test_df.drop(['PassengerId'], axis=1)

    # Load the trained pipeline if not provided
    if pipeline is None:
        print("Loading trained pipeline...")
        pipeline = joblib.load('data_out/titanic_model.joblib')

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
    submission_df.to_csv('data_out/submission.csv', index=False)
    print("Done! Predictions saved to data_out/submission.csv")

if __name__ == "__main__":
    # Train the model
    pipeline = train_model()
    
    # Make predictions
    make_predictions(pipeline) 