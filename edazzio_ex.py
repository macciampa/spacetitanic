# https://www.kaggle.com/code/elainedazzio/20250522-st-ensemble?scriptVersionId=241870633

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import boxcox
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from functools import partial
from copy import deepcopy
import gc
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility across all components
RANDOM_STATE = 2023
np.random.seed(RANDOM_STATE)

# GPU setup
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# =============================================================================
def load_and_explore_data():
    """
    Load training and test datasets and perform basic exploration.
    
    Returns:
        tuple: (train_df, test_df) - pandas DataFrames
    """
    train = pd.read_csv("../data_in/train.csv")
    test = pd.read_csv("../data_in/test.csv")
    
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nTarget distribution:")
    print(train['Transported'].value_counts(normalize=True))
    
    return train, test

# =============================================================================
# 3. DATA CLEANING AND PREPROCESSING
# =============================================================================
def extract_passenger_features(df):
    """
    Extract structured features from PassengerId and Cabin fields.
    
    PassengerId format: gggg_pp (group_passenger)
    Cabin format: deck/num/side
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with new features: group, cabin_deck, cabin_num, cabin_side, Last_Name
    """
    # Extract group ID from PassengerId - passengers in same group may have similar behavior
    df['group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    
    # Parse cabin information - deck and side may indicate ship location preferences
    df[['cabin_deck', 'cabin_num', 'cabin_side']] = df['Cabin'].str.split('/', expand=True)
    df['cabin_num'] = pd.to_numeric(df['cabin_num'], errors='coerce')  # Convert to numeric for potential ordering
    
    # Extract last name for family relationship analysis
    df['Last_Name'] = df['Name'].str.split().str[-1].str.lower()
    
    return df

def intelligent_missing_value_handling(train, test):
    """
    Handle missing values using domain knowledge about the Spaceship Titanic scenario.
    
    Key insight: Passengers in CryoSleep cannot spend money on amenities, creating
    a logical relationship between expenditure and CryoSleep status.
    
    Args:
        train, test: DataFrames to process
        
    Returns:
        tuple: (train, test) with missing values handled and new features created
    """
    
    # Define expenditure features for logical reasoning
    expenditure_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # First, handle missing values in expenditure features with 0 (assuming no spending = 0)
    for col in expenditure_features:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)
    
    # Calculate total expenditure - key feature for CryoSleep detection
    train["Expenditure"] = train[expenditure_features].sum(axis=1)
    test["Expenditure"] = test[expenditure_features].sum(axis=1)
    
    # IMPROVED LOGIC: Use actual CryoSleep values when available
    # First, create inference based on expenditure
    train['CryoSleep_inferred'] = np.where(train['Expenditure'] == 0, True, False)
    test['CryoSleep_inferred'] = np.where(test['Expenditure'] == 0, True, False)
    
    # Use actual CryoSleep where available, else use inference
    if 'CryoSleep' in train.columns:
        train['CryoSleep'] = train['CryoSleep'].fillna(train['CryoSleep_inferred']).astype(float)
    else:
        train['CryoSleep'] = train['CryoSleep_inferred'].astype(float)
        
    if 'CryoSleep' in test.columns:
        test['CryoSleep'] = test['CryoSleep'].fillna(test['CryoSleep_inferred']).astype(float)
    else:
        test['CryoSleep'] = test['CryoSleep_inferred'].astype(float)
    
    # Drop the temporary inference column
    train.drop('CryoSleep_inferred', axis=1, inplace=True)
    test.drop('CryoSleep_inferred', axis=1, inplace=True)
    
    # Handle VIP column similarly
    if 'VIP' in train.columns:
        # Fill missing VIP values based on CryoSleep relationship
        train.loc[train['VIP'].isna(), 'VIP'] = ~train.loc[train['VIP'].isna(), 'CryoSleep'].astype(bool)
        test.loc[test['VIP'].isna(), 'VIP'] = ~test.loc[test['VIP'].isna(), 'CryoSleep'].astype(bool)
    else:
        train['VIP'] = np.where(train['CryoSleep'] == 0, 1, 0)
        test['VIP'] = np.where(test['CryoSleep'] == 0, 1, 0)
    
    # Enforce logical consistency: CryoSleep passengers have zero expenditure
    for col in expenditure_features:
        train.loc[train['CryoSleep'] == 1, col] = 0
        test.loc[test['CryoSleep'] == 1, col] = 0
    
    # Convert boolean columns to numeric for model compatibility
    bool_cols = ['CryoSleep', 'VIP']
    for col in bool_cols:
        train[col] = train[col].astype(float)
        test[col] = test[col].astype(float)
    
    # Convert target to numeric
    train['Transported'] = train['Transported'].astype(int)
    
    # Use KNN imputation for remaining numeric missing values
    # KNN preserves relationships between features better than mean/median imputation
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Transported']
    
    if any(train[numeric_cols].isnull().sum() > 0):
        imputer = KNNImputer(n_neighbors=5)
        train[numeric_cols] = imputer.fit_transform(train[numeric_cols])
        test[numeric_cols] = imputer.transform(test[numeric_cols])
    
    # Handle categorical missing values with mode (most frequent value)
    categorical_cols = ['HomePlanet', 'Destination', 'cabin_deck', 'cabin_side']
    for col in categorical_cols:
        if col in train.columns:
            mode_val = train[col].mode()[0] if not train[col].mode().empty else 'Unknown'
            train[col].fillna(mode_val, inplace=True)
            test[col].fillna(mode_val, inplace=True)
    
    return train, test

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
def create_expenditure_features(train, test):
    """
    Create additional expenditure-based features beyond the individual amounts.
    
    These features capture spending patterns and luxury preferences.
    
    Args:
        train, test: DataFrames to enhance
        
    Returns:
        tuple: (train, test) with new expenditure features
    """
    expenditure_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Total expenditure - captures overall spending level
    train['expenditure'] = train[expenditure_features].sum(axis=1)
    test['expenditure'] = test[expenditure_features].sum(axis=1)
    
    # Luxury spending - combines high-end amenities (Spa + VRDeck)
    # May indicate passenger class or lifestyle preferences
    train['luxury_spending'] = train['Spa'] + train['VRDeck']
    test['luxury_spending'] = test['Spa'] + test['VRDeck']
    
    # Group size only - NO TARGET LEAKAGE
    # Calculate from combined train+test to ensure all groups are counted
    all_data = pd.concat([train[['group']], test[['group']]], axis=0)
    group_size = all_data.groupby('group').size().to_dict()
    train['group_size'] = train['group'].map(group_size)
    test['group_size'] = test['group'].map(group_size)
    
    # Same cabin count - passengers sharing exact cabin location
    # This is leakage-free as it doesn't use target information
    train['cabin_full'] = train['cabin_deck'].astype(str) + '_' + train['cabin_num'].astype(str) + '_' + train['cabin_side'].astype(str)
    test['cabin_full'] = test['cabin_deck'].astype(str) + '_' + test['cabin_num'].astype(str) + '_' + test['cabin_side'].astype(str)
    
    all_cabins = pd.concat([train[['cabin_full']], test[['cabin_full']]], axis=0)
    cabin_counts = all_cabins.groupby('cabin_full').size().to_dict()
    train['cabin_companions'] = train['cabin_full'].map(cabin_counts) - 1  # Subtract 1 to exclude self
    test['cabin_companions'] = test['cabin_full'].map(cabin_counts) - 1
    
    # Drop temporary cabin_full column
    train.drop('cabin_full', axis=1, inplace=True)
    test.drop('cabin_full', axis=1, inplace=True)
    
    # CryoSleep interaction features
    # These capture how CryoSleep status affects other relationships
    for col in ['Age', 'group_size', 'cabin_companions']:
        if col in train.columns:
            train[f'CryoSleep_x_{col}'] = train['CryoSleep'] * train[col]
            test[f'CryoSleep_x_{col}'] = test['CryoSleep'] * test[col]
    
    # Age groups - children, teens, young adults, middle-aged, elderly
    # These groups may have different transport patterns
    train['Age_group'] = pd.cut(train['Age'], bins=[0, 12, 18, 30, 50, 100], labels=[0, 1, 2, 3, 4])
    test['Age_group'] = pd.cut(test['Age'], bins=[0, 12, 18, 30, 50, 100], labels=[0, 1, 2, 3, 4])
    
    # Fill NaN age groups with a separate category
    train['Age_group'] = train['Age_group'].cat.add_categories([5])
    test['Age_group'] = test['Age_group'].cat.add_categories([5])
    train['Age_group'] = train['Age_group'].fillna(5)
    test['Age_group'] = test['Age_group'].fillna(5)
    
    train['Age_group'] = train['Age_group'].astype(int)
    test['Age_group'] = test['Age_group'].astype(int)
    
    # Solo traveler indicator
    train['is_solo'] = ((train['group_size'] == 1) & (train['cabin_companions'] == 0)).astype(int)
    test['is_solo'] = ((test['group_size'] == 1) & (test['cabin_companions'] == 0)).astype(int)
    
    # Cabin deck ordering and location features
    # Deck order mapping (assuming alphabetical = vertical order)
    deck_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    train['deck_level'] = train['cabin_deck'].map(deck_order).fillna(0)
    test['deck_level'] = test['cabin_deck'].map(deck_order).fillna(0)
    
    # Cabin side encoding (P=Port=0, S=Starboard=1)
    side_map = {'P': 0, 'S': 1}
    train['cabin_side_num'] = train['cabin_side'].map(side_map).fillna(-1)
    test['cabin_side_num'] = test['cabin_side'].map(side_map).fillna(-1)
    
    # Cabin section features (front/middle/back of ship)
    # Assuming cabin numbers indicate position along the ship
    train['cabin_section'] = pd.cut(train['cabin_num'], bins=[-1, 300, 600, 900, 2000], labels=[0, 1, 2, 3])
    test['cabin_section'] = pd.cut(test['cabin_num'], bins=[-1, 300, 600, 900, 2000], labels=[0, 1, 2, 3])
    
    train['cabin_section'] = train['cabin_section'].cat.add_categories([4])
    test['cabin_section'] = test['cabin_section'].cat.add_categories([4])
    train['cabin_section'] = train['cabin_section'].fillna(4)
    test['cabin_section'] = test['cabin_section'].fillna(4)
    
    train['cabin_section'] = train['cabin_section'].astype(int)
    test['cabin_section'] = test['cabin_section'].astype(int)
    
    # Distance from center features
    # Upper decks + middle section might be premium locations
    train['deck_distance_from_middle'] = np.abs(train['deck_level'] - 4)
    test['deck_distance_from_middle'] = np.abs(test['deck_level'] - 4)
    
    # Premium cabin indicator (upper decks + middle section)
    train['premium_cabin'] = ((train['deck_level'] <= 3) & (train['cabin_section'] == 1)).astype(int)
    test['premium_cabin'] = ((test['deck_level'] <= 3) & (test['cabin_section'] == 1)).astype(int)
    
    return train, test

def apply_numerical_transformations(train, test, columns):
    """
    Apply and validate numerical transformations, keeping only those that improve performance.
    
    This function tests log and square root transformations using cross-validation
    and only retains transformations that demonstrably improve single-feature performance.
    
    Args:
        train, test: DataFrames to transform
        columns: List of column names to test transformations on
        
    Returns:
        tuple: (train, test) with optimal transformations applied
    """
    
    def find_optimal_cutoff(y_true, y_pred_proba):
        """Find threshold that maximizes accuracy for binary classification."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        accuracy_scores = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            accuracy_scores.append(accuracy_score(y_true, y_pred))
        optimal_idx = np.argmax(accuracy_scores)
        return thresholds[optimal_idx]
    
    def test_transformation_performance(X, y, feature_name):
        """
        Test single-feature performance using cross-validation.
        
        Uses logistic regression as a simple, unbiased test of feature quality.
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Simple logistic regression test
            model = LogisticRegression(random_state=RANDOM_STATE)
            model.fit(X_train_fold.values.reshape(-1, 1), y_train_fold)
            y_pred_proba = model.predict_proba(X_val_fold.values.reshape(-1, 1))[:, 1]
            
            cutoff = find_optimal_cutoff(y_val_fold, y_pred_proba)
            y_pred = (y_pred_proba >= cutoff).astype(int)
            scores.append(accuracy_score(y_val_fold, y_pred))
        
        return np.mean(scores)
    
    # Test transformations for each specified column
    for col in columns:
        if col in train.columns:
            # Baseline performance with original feature
            original_score = test_transformation_performance(train[col], train['Transported'], col)
            
            # Test different transformations
            transformations = {}
            
            # Log transformation (handles zero values with log1p)
            train[f'log_{col}'] = np.log1p(train[col])
            test[f'log_{col}'] = np.log1p(test[col])
            transformations[f'log_{col}'] = test_transformation_performance(
                train[f'log_{col}'], train['Transported'], f'log_{col}'
            )
            
            # Square root transformation
            train[f'sqrt_{col}'] = np.sqrt(train[col])
            test[f'sqrt_{col}'] = np.sqrt(test[col])
            transformations[f'sqrt_{col}'] = test_transformation_performance(
                train[f'sqrt_{col}'], train['Transported'], f'sqrt_{col}'
            )
            
            # Keep only the best transformation (if it beats original)
            best_feature = max(transformations.items(), key=lambda x: x[1])
            if best_feature[1] > original_score:
                print(f"Best transformation for {col}: {best_feature[0]} (score: {best_feature[1]:.4f})")
                # Remove the transformation we're not keeping
                for trans_name in transformations.keys():
                    if trans_name != best_feature[0]:
                        train.drop(columns=[trans_name], inplace=True)
                        test.drop(columns=[trans_name], inplace=True)
            else:
                # Original is best - remove all transformations
                for trans_name in transformations.keys():
                    train.drop(columns=[trans_name], inplace=True)
                    test.drop(columns=[trans_name], inplace=True)
    
    return train, test

def encode_categorical_features(train, test, categorical_features):
    """
    Apply one-hot encoding to categorical features.
    
    REMOVED target encoding to reduce overfitting.
    
    Args:
        train, test: DataFrames to encode
        categorical_features: List of categorical column names
        
    Returns:
        tuple: (train, test) with encoded features and original categoricals removed
    """
    
    for feature in categorical_features:
        if feature in train.columns:
            # One-hot encode
            dummies_train = pd.get_dummies(train[feature], prefix=feature, dummy_na=False)
            dummies_test = pd.get_dummies(test[feature], prefix=feature, dummy_na=False)
            
            # Add to dataframes
            train = pd.concat([train, dummies_train], axis=1)
            test = pd.concat([test, dummies_test], axis=1)
            
            # Ensure test has all columns from train (fill with 0 if missing)
            for col in dummies_train.columns:
                if col not in test.columns:
                    test[col] = 0
                    
            # Ensure train has all columns from test (though this is rare)
            for col in dummies_test.columns:
                if col not in train.columns:
                    train[col] = 0
            
            # Remove original categorical feature
            train.drop(columns=[feature], inplace=True)
            test.drop(columns=[feature], inplace=True)
    
    return train, test

def create_text_features(train, test):
    """
    Extract features from text fields using TF-IDF and dimensionality reduction.
    
    Process: Name -> TF-IDF -> SVD -> 5 numerical features
    This captures family relationships and name patterns efficiently.
    
    Args:
        train, test: DataFrames to process
        
    Returns:
        tuple: (train, test) with text features converted to numerical features
    """
    if 'Last_Name' in train.columns:
        # Apply TF-IDF vectorization to last names
        # This captures character-level patterns that might indicate family groups
        vectorizer = TfidfVectorizer(max_features=1000, lowercase=True)
        
        # Fit on combined data to ensure consistency
        all_names = pd.concat([train['Last_Name'], test['Last_Name']], axis=0)
        vectorizer.fit(all_names.fillna('unknown'))
        
        # Transform both datasets
        train_tfidf = vectorizer.transform(train['Last_Name'].fillna('unknown'))
        test_tfidf = vectorizer.transform(test['Last_Name'].fillna('unknown'))
        
        # Apply dimensionality reduction to avoid curse of dimensionality
        # SVD reduces 1000 TF-IDF features to 5 meaningful components
        svd = TruncatedSVD(n_components=5, random_state=RANDOM_STATE)
        train_reduced = svd.fit_transform(train_tfidf)
        test_reduced = svd.transform(test_tfidf)
        
        # Add reduced features to dataframes
        for i in range(5):
            train[f'name_tfidf_{i}'] = train_reduced[:, i]
            test[f'name_tfidf_{i}'] = test_reduced[:, i]
        
        # Clean up original text columns
        train.drop(columns=['Name', 'Last_Name'], inplace=True, errors='ignore')
        test.drop(columns=['Name', 'Last_Name'], inplace=True, errors='ignore')
    
    return train, test

def create_polynomial_features(X_train, X_test, y_train):
    """
    Create polynomial interaction features for the most important features.
    
    This helps capture non-linear relationships between top features.
    
    Args:
        X_train, X_test: Training and test features
        y_train: Training target
        
    Returns:
        tuple: (X_train, X_test) with polynomial features added
    """
    print("\n8.5. Creating polynomial features for top interactions...")
    
    # Quick feature importance check
    quick_model = xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    quick_model.fit(X_train, y_train)
    
    # Get top 10 most important features
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': quick_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    print("\nTop 10 features by importance:")
    for idx, row in importances.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Create polynomial features for top 5
    top_features = importances['feature'].head(5).tolist()
    print(f"\nCreating polynomial interactions for: {top_features}")
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    
    X_train_poly = poly.fit_transform(X_train[top_features])
    X_test_poly = poly.transform(X_test[top_features])
    
    # Get feature names
    poly_feature_names = poly.get_feature_names_out(top_features)
    
    # Add only the interaction features (skip the original features)
    new_features_count = 0
    for i, name in enumerate(poly_feature_names[len(top_features):]):  # Skip original features
        col_name = f'poly_{name}'
        X_train[col_name] = X_train_poly[:, len(top_features) + i]
        X_test[col_name] = X_test_poly[:, len(top_features) + i]
        new_features_count += 1
    
    print(f"Added {new_features_count} polynomial interaction features")
    
    # NEW: Additional CryoSleep interactions with key features
    print("\nAdding targeted CryoSleep interactions...")
    
    # Manual CryoSleep interactions with features not already in polynomial set
    cryo_interact_features = ['deck_level', 'cabin_side_num', 'VIP', 'Age_group', 'is_solo']
    interaction_count = 0
    
    for feat in cryo_interact_features:
        if feat in X_train.columns and f'poly_CryoSleep {feat}' not in X_train.columns:
            X_train[f'CryoSleep_x_{feat}_manual'] = X_train['CryoSleep'] * X_train[feat]
            X_test[f'CryoSleep_x_{feat}_manual'] = X_test['CryoSleep'] * X_test[feat]
            interaction_count += 1
    
    # CryoSleep + HomePlanet specific combinations
    if 'HomePlanet_Earth' in X_train.columns:
        X_train['CryoSleep_Earth'] = (X_train['CryoSleep'] == 1) & (X_train['HomePlanet_Earth'] == 1)
        X_test['CryoSleep_Earth'] = (X_test['CryoSleep'] == 1) & (X_test['HomePlanet_Earth'] == 1)
        X_train['CryoSleep_Earth'] = X_train['CryoSleep_Earth'].astype(int)
        X_test['CryoSleep_Earth'] = X_test['CryoSleep_Earth'].astype(int)
        interaction_count += 1
    
    if 'HomePlanet_Europa' in X_train.columns:
        X_train['CryoSleep_Europa'] = (X_train['CryoSleep'] == 1) & (X_train['HomePlanet_Europa'] == 1)
        X_test['CryoSleep_Europa'] = (X_test['CryoSleep'] == 1) & (X_test['HomePlanet_Europa'] == 1)
        X_train['CryoSleep_Europa'] = X_train['CryoSleep_Europa'].astype(int)
        X_test['CryoSleep_Europa'] = X_test['CryoSleep_Europa'].astype(int)
        interaction_count += 1
    
    # CryoSleep + Deck specific combinations
    for deck in ['cabin_deck_B', 'cabin_deck_C', 'cabin_deck_D', 'cabin_deck_E', 'cabin_deck_F', 'cabin_deck_G']:
        if deck in X_train.columns:
            X_train[f'CryoSleep_{deck}'] = (X_train['CryoSleep'] == 1) & (X_train[deck] == 1)
            X_test[f'CryoSleep_{deck}'] = (X_test['CryoSleep'] == 1) & (X_test[deck] == 1)
            X_train[f'CryoSleep_{deck}'] = X_train[f'CryoSleep_{deck}'].astype(int)
            X_test[f'CryoSleep_{deck}'] = X_test[f'CryoSleep_{deck}'].astype(int)
            interaction_count += 1
    
    print(f"Added {interaction_count} manual CryoSleep interaction features")
    
    return X_train, X_test

# =============================================================================
# 5. MODEL CLASSES AND ENSEMBLE
# =============================================================================
class OptimizedClassifier:
    """
    Wrapper class for gradient boosting models with REDUCED regularization.
    
    Less aggressive parameters to improve generalization.
    """
    
    def __init__(self, n_estimators=1000, random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # XGBoost: Balanced parameters
        self.xgb_params = {
            'colsample_bytree': 0.8,
            'learning_rate': 0.05,       # Increased from 0.02
            'max_depth': 5,              # Increased from 4
            'n_estimators': n_estimators,
            'subsample': 0.8,
            'objective': 'binary:logistic',
            'random_state': random_state,
            'n_jobs': -1,
            'tree_method': 'gpu_hist' if str(device) == 'cuda' else 'auto'
        }
        
        # LightGBM: Reduced regularization
        self.lgb_params = {
            'colsample_bytree': 0.8,
            'learning_rate': 0.05,       # Increased from 0.008
            'max_depth': 6,              # Increased from 5
            'n_estimators': n_estimators,
            'reg_alpha': 0.1,            # Reduced from 0.14
            'reg_lambda': 0.1,           # GREATLY reduced from 0.93
            'subsample': 0.8,            # Increased from 0.62
            'objective': 'binary',
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1,
            'device': 'gpu' if str(device) == 'cuda' else 'cpu'
        }
        
        # CatBoost: Standard parameters
        self.cat_params = {
            'learning_rate': 0.05,       # Increased from 0.01
            'depth': 6,                  # Increased from 5
            'iterations': n_estimators,
            'random_state': random_state,
            'verbose': False,
            'task_type': 'GPU' if str(device) == 'cuda' else 'CPU'
        }
    
    def get_models(self):
        """Return dictionary of configured model instances."""
        return {
            'xgb': xgb.XGBClassifier(**self.xgb_params),
            'lgb': lgb.LGBMClassifier(**self.lgb_params),
            'cat': CatBoostClassifier(**self.cat_params)
        }

class OptunaEnsemble:
    """
    Ensemble class that optimizes model weights using Optuna.
    
    This finds the optimal combination of model predictions to maximize accuracy.
    """
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.weights = None
        self.study = None
    
    def _objective(self, trial, y_true, y_preds):
        """
        Objective function for Optuna optimization.
        
        Optimizes both ensemble weights and finds the best accuracy achievable.
        """
        # Define weights for each model's predictions
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]
        
        # Calculate weighted average prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)
        
        # Find optimal threshold for this weighted prediction
        fpr, tpr, thresholds = roc_curve(y_true, weighted_pred)
        accuracy_scores = []
        for threshold in thresholds:
            y_pred_binary = (weighted_pred >= threshold).astype(int)
            accuracy_scores.append(accuracy_score(y_true, y_pred_binary))
        
        return max(accuracy_scores)
    
    def fit(self, y_true, y_preds, n_trials=1000):
        """
        Fit ensemble weights using Optuna optimization.
        
        Args:
            y_true: True labels
            y_preds: List of model predictions
            n_trials: Number of optimization trials
        """
        optuna.logging.set_verbosity(optuna.logging.ERROR)  # Suppress output
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, direction='maximize')
        
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        
        # Store optimal weights
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]
        return self
    
    def predict(self, y_preds):
        """Make predictions using optimized weights."""
        if self.weights is None:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

# =============================================================================
# 6. TRAINING AND EVALUATION
# =============================================================================
def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find threshold that maximizes accuracy for binary classification.
    
    This is crucial for competition performance as the metric is accuracy, not AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    accuracy_scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        accuracy_scores.append(accuracy_score(y_true, y_pred))
    optimal_idx = np.argmax(accuracy_scores)
    return thresholds[optimal_idx]

def train_and_evaluate_models(X_train, y_train, X_test):
    """
    Train ensemble models with cross-validation and return predictions.
    
    Uses 10-fold stratified cross-validation to ensure robust evaluation and
    proper out-of-fold predictions for test set.
    
    Args:
        X_train, y_train: Training features and target
        X_test: Test features
        
    Returns:
        tuple: (test_predictions, mean_cv_score)
    """
    
    # Cross-validation setup - 10 folds for robust evaluation
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    # Initialize results storage
    test_predictions = np.zeros(X_test.shape[0])
    cv_scores = []
    
    # Initialize classifier with pre-tuned models
    classifier = OptimizedClassifier()
    models = classifier.get_models()
    
    print("Starting cross-validation training...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Split data for this fold
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train individual models
        fold_oof_preds = []  # Out-of-fold predictions for ensemble optimization
        fold_test_preds = [] # Test predictions for this fold
        
        for name, model in models.items():
            # Train with early stopping for gradient boosting models
            if name == 'xgb':
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=100,  # Stop if no improvement for 100 rounds
                    verbose=False
                )
            elif name == 'lgb':
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
            elif name == 'cat':
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=100,
                    verbose=False
                )
            else:
                # For models without early stopping
                model.fit(X_train_fold, y_train_fold)
            
            # Generate predictions
            val_pred = model.predict_proba(X_val_fold)[:, 1]  # Probability of class 1
            test_pred = model.predict_proba(X_test)[:, 1]
            
            fold_oof_preds.append(val_pred)
            fold_test_preds.append(test_pred)
            
            # Calculate individual model performance for monitoring
            threshold = find_optimal_threshold(y_val_fold, val_pred)
            val_pred_binary = (val_pred >= threshold).astype(int)
            score = accuracy_score(y_val_fold, val_pred_binary)
            print(f"  {name}: {score:.4f}")
        
        # Optimize ensemble weights for this fold using Optuna
        ensemble = OptunaEnsemble(random_state=RANDOM_STATE)
        ensemble.fit(y_val_fold.values, fold_oof_preds, n_trials=500)
        
        # Evaluate ensemble performance
        ensemble_val_pred = ensemble.predict(fold_oof_preds)
        threshold = find_optimal_threshold(y_val_fold, ensemble_val_pred)
        ensemble_val_binary = (ensemble_val_pred >= threshold).astype(int)
        ensemble_score = accuracy_score(y_val_fold, ensemble_val_binary)
        
        print(f"  Ensemble: {ensemble_score:.4f}")
        cv_scores.append(ensemble_score)
        
        # Add ensemble test predictions for this fold
        ensemble_test_pred = ensemble.predict(fold_test_preds)
        test_predictions += ensemble_test_pred / n_splits
        
        # Clean up memory
        gc.collect()
    
    # Calculate final cross-validation statistics
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"\nCross-validation results:")
    print(f"Mean CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    
    return test_predictions, mean_cv_score

# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================
def main():
    """
    Main execution pipeline that orchestrates the entire machine learning workflow.
    
    This function represents the complete pipeline from data loading to submission creation,
    following best practices for competition machine learning.
    """
    
    print("=== SPACESHIP TITANIC COMPETITION PIPELINE ===\n")
    
    # Step 1: Load and explore data
    print("1. Loading data...")
    train, test = load_and_explore_data()
    
    # Step 2: Extract structured features from complex fields
    print("\n2. Extracting basic features...")
    train = extract_passenger_features(train)
    test = extract_passenger_features(test)
    
    # Step 3: Handle missing values using domain knowledge
    print("\n3. Handling missing values...")
    train, test = intelligent_missing_value_handling(train, test)
    
    # Step 4: Create derived expenditure features
    print("\n4. Creating expenditure features...")
    train, test = create_expenditure_features(train, test)
    
    # Step 5: Test and apply numerical transformations
    print("\n5. Applying numerical transformations...")
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    train, test = apply_numerical_transformations(train, test, numerical_cols)
    
    # Step 6: Encode categorical features
    print("\n6. Encoding categorical features...")
    categorical_cols = ['HomePlanet', 'Destination', 'cabin_deck', 'cabin_side']
    train, test = encode_categorical_features(train, test, categorical_cols)
    
    # Step 7: Convert text to numerical features
    print("\n7. Creating text features...")
    train, test = create_text_features(train, test)
    
    # Step 8: Prepare final datasets for modeling
    print("\n8. Preparing final datasets...")
    
    # Store PassengerId for submission file
    passenger_ids = test['PassengerId'].copy()
    
    # Remove non-feature columns
    columns_to_drop = ['PassengerId', 'Cabin']
    train.drop(columns=[col for col in columns_to_drop if col in train.columns], inplace=True)
    test.drop(columns=[col for col in columns_to_drop if col in test.columns], inplace=True)
    
    # Separate features and target
    X_train = train.drop('Transported', axis=1)
    y_train = train['Transported']
    X_test = test.copy()
    
    print(f"Initial training shape: {X_train.shape}")
    print(f"Initial test shape: {X_test.shape}")
    
    # Step 8.5: Create polynomial features for top interactions
    X_train, X_test = create_polynomial_features(X_train, X_test, y_train)
    
    print(f"Final training shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")
    
    # Step 9: Train models and generate predictions
    print("\n9. Training models and making predictions...")
    test_predictions, cv_score = train_and_evaluate_models(X_train, y_train, X_test)
    
    # Step 10: Create submission file
    print("\n10. Creating submission...")
    
    # Apply fixed threshold of 0.5
    final_predictions = (test_predictions >= 0.5).astype(bool)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': final_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    
    # Final summary
    print(f"\nPipeline completed!")
    print(f"CV Score: {cv_score:.4f}")
    print(f"Submission saved as 'submission.csv'")
    print(f"Predicted Transported rate: {final_predictions.mean():.3f}")
    
    return submission

# =============================================================================
# 8. EXECUTION
# =============================================================================
if __name__ == "__main__":
    submission = main()
