import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
import os
from visualize import create_deck_side_heatmap, create_age_survival_histogram, create_destination_homeplanet_heatmap, create_spending_survival_histogram, create_age_transported_distribution, create_cabin_num_survival_histogram
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')

# Debug flag
DEBUG = True

# Visualization flag
VISUALIZE = True

# Stats flag
STATS = False

# Create output directory if it doesn't exist
os.makedirs('data_out', exist_ok=True)

def engineer_features(df):
    """Apply feature engineering to the dataset"""
    # Custom imputer for Cabin: fill missing with mode among same group
    # df['Group'] = df['PassengerId'].str.split('_').str[0]
    # mask_missing_cabin = df['Cabin'].isnull()
    # for idx in df[mask_missing_cabin].index:
    #     group = df.at[idx, 'Group']
    #     candidates = df[(df['Group'] == group) & (~df['Cabin'].isnull())]['Cabin']
    #     if not candidates.empty:
    #         mode = candidates.mode().iloc[0]
    #         df.at[idx, 'Cabin'] = mode
    # df.drop(columns=['Group'], inplace=True)
    
    # Custom imputer for HomePlanet: fill missing with mode among same last name
    # df['LastName'] = df['Name'].fillna('Unknown').str.split(' ').str[-1]
    # mask_missing = df['HomePlanet'].isnull()
    # for idx in df[mask_missing].index:
    #     last_name = df.at[idx, 'LastName']
    #     if last_name == "Unknown":
    #         continue  # Skip imputation for missing names
    #     candidates = df[(df['LastName'] == last_name) & (~df['HomePlanet'].isnull())]['HomePlanet']
    #     if not candidates.empty:
    #         mode = candidates.mode().iloc[0]
    #         df.at[idx, 'HomePlanet'] = mode
    # df.drop(columns=['LastName'], inplace=True)
    
    # Create Infant feature
    df['Infant'] = df['Age'].fillna(-1) <= 4

    # Create TotalSpent feature
    spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpent'] = df[spending_columns].fillna(0).sum(axis=1)

    # Create NoMoney feature
    df['NoMoney'] = df['TotalSpent'] == 0

    # Split Cabin into Deck and Side features
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Side'] = df['Side'].fillna('Unknown')  # Handle missing values
    df['Deck'] = df['Deck'].fillna('Unknown')  # Handle missing values
    
    # Create StarboardBC feature
    df['StarboardBC'] = (df['Deck'].isin(['B', 'C'])) & (df['Side'] == 'S')
    
    # Create Route feature (combines HomePlanet and Destination)
    df['Route'] = df['HomePlanet'] + '_to_' + df['Destination']

    # Create Group and GroupSize features
    # df['Group'] = df['PassengerId'].str.split('_').str[0]
    # group_sizes = df.groupby('Group')['PassengerId'].transform('count')
    # df['GroupSize'] = group_sizes
    # df['TravellingSolo'] = df['GroupSize'] == 1
    # Remove temporary Group column
    # df.drop(columns=['Group'], inplace=True)
    
    # CabinNumRegion feature: bin Num into regions of size 100
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')
    min_num = int(df['Num'].min())
    max_num = int(df['Num'].max())
    bin_edges = list(range(min_num, max_num + 101, 100))
    df['CabinNumRegion'] = pd.cut(df['Num'], bins=bin_edges)
    
    return df

def drop_unused_features(df, is_training=False):
    """Drop the same features in both training and prediction. Drops 'Transported' only in training."""
    drop_cols = ['PassengerId', 'Name', 'Cabin', 'Num']
    if is_training:
        drop_cols = ['Transported'] + drop_cols
    drop_cols = [col for col in drop_cols if col in df.columns]
    return df.drop(columns=drop_cols)

def train_model():
    # Load the data
    print("Loading training data...")
    df = pd.read_csv('data_in/train.csv')

    # Apply feature engineering
    df = engineer_features(df)

    # Create deck/side heatmap visualization
    if VISUALIZE:
        create_deck_side_heatmap(df, stats=STATS)
        create_age_survival_histogram(df, stats=STATS)
        create_destination_homeplanet_heatmap(df, stats=STATS)
        create_spending_survival_histogram(df, stats=STATS)
        create_age_transported_distribution(df)
        create_cabin_num_survival_histogram(df, stats=STATS)

    # Display basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

    # Separate features and target (drop unused features here)
    y = df['Transported']
    X = drop_unused_features(df, is_training=True)

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

    # Fit pipeline and check for under/overfitting
    pipeline.fit(X, y)
    train_score = pipeline.score(X, y)
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    cv_mean = cv_scores.mean()
    print(f"\nTraining score: {train_score:.4f}")
    print(f"Cross-validation mean score: {cv_mean:.4f}")
    print(f"Cross-validation scores: {cv_scores}")

    # Get feature names after preprocessing
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Save feature names to text file
    if DEBUG:
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': pipeline.named_steps['classifier'].feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance['rank'] = range(1, len(feature_importance) + 1)
        
        with open('data_out/feature_names.txt', 'w') as f:
            f.write("All feature names after preprocessing:\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Rank':<6} {'Feature':<50} {'Importance':<12}\n")
            f.write("-" * 70 + "\n")
            for _, row in feature_importance.iterrows():
                f.write(f"{row['rank']:<6} {row['feature']:<50} {row['importance']:<12.6f}\n")
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

    # Apply feature engineering
    test_df = engineer_features(test_df)

    # Save PassengerId for submission
    passenger_ids = test_df['PassengerId']

    # Prepare features (same as in training)
    X_test = drop_unused_features(test_df, is_training=False)

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