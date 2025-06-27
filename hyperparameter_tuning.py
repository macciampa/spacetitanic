import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
import catboost as cb
import joblib
import os
from titanic_model import engineer_features, drop_unused_features

# Load and prepare data
print("Loading and preparing data...")
df = pd.read_csv('data_in/train.csv')
df = engineer_features(df)
y = df['Transported']
X = drop_unused_features(df, is_training=True)

# Preprocessing pipeline (same as model_comparison.py)
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 
                   'Infant', 'NoMoney', 'StarboardBC', 'Route']
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent']

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Top models by accuracy (excluding VotingClassifier)
model_info = [
    ('CatBoostClassifier', cb.CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False)),
    ('LGBMClassifier', lgb.LGBMClassifier(random_state=42, verbose=-1)),
    ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=42)),
    ('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('RandomForestClassifier', RandomForestClassifier(random_state=42)),
    ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
]

# Define parameter grids for each model
param_grids = {
    'CatBoostClassifier': {
        'classifier__iterations': [100, 200],
        'classifier__depth': [4, 6, 8],
        'classifier__learning_rate': [0.05, 0.1, 0.2]
    },
    'LGBMClassifier': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [4, 6, 8],
        'classifier__learning_rate': [0.05, 0.1, 0.2]
    },
    'GradientBoostingClassifier': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2]
    },
    'XGBClassifier': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2]
    },
    'RandomForestClassifier': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, 15]
    },
    'LogisticRegression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
}

# Output directory
os.makedirs('data_out/hyperparameter_tuning', exist_ok=True)

results = []

for name, model in model_info:
    print(f"\nTuning {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    param_grid = param_grids.get(name, {})
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    search.fit(X, y)
    best_score = search.best_score_
    best_params = search.best_params_
    best_estimator = search.best_estimator_
    print(f"Best score for {name}: {best_score:.4f}")
    print(f"Best params: {best_params}")
    results.append((name, best_score, best_params))
    # Save the best model
    joblib.dump(best_estimator, f'data_out/hyperparameter_tuning/best_{name}.joblib')

# Save summary
results_df = pd.DataFrame(results, columns=['Model', 'Best_Score', 'Best_Params'])
results_df.to_csv('data_out/hyperparameter_tuning/hyperparameter_tuning_results.csv', index=False)
print("\nHyperparameter tuning completed. Results saved to data_out/hyperparameter_tuning/hyperparameter_tuning_results.csv") 