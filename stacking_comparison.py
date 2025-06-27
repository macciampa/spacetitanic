import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
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
top_model_names = [
    'CatBoostClassifier',
    'LGBMClassifier',
    'GradientBoostingClassifier',
    'XGBClassifier',
    'RandomForestClassifier',
    'LogisticRegression',
]

# Load best tuned models (classifiers only)
best_classifiers = []
for name in top_model_names:
    if name == 'LGBMClassifier':
        pipeline = joblib.load(f'data_out/hyperparameter_tuning/best_{name}.joblib')
        clf = pipeline.named_steps['classifier']
        # Set verbose=-1 if not already set
        if hasattr(clf, 'set_params'):
            clf.set_params(verbose=-1)
        best_classifiers.append((name, clf))
    else:
        pipeline = joblib.load(f'data_out/hyperparameter_tuning/best_{name}.joblib')
        clf = pipeline.named_steps['classifier']
        best_classifiers.append((name, clf))

# Output directory
os.makedirs('data_out/stacking_comparison', exist_ok=True)

results = []

# Try stacking with top 3, 4, 5, and 6 models
for n in range(3, 7):
    estimators = best_classifiers[:n]
    print(f"\nEvaluating stacking with top {n} models: {[name for name, _ in estimators]}")
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )
    stacking_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('stack', stacking_clf)
    ])
    scores = cross_val_score(stacking_pipeline, X, y, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"Stacking (top {n}) CV mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")
    results.append((n, [name for name, _ in estimators], mean_score, std_score, scores))
    # Save the best model (fit on all data) if it's the best so far
    if mean_score == max(r[2] for r in results):
        stacking_pipeline.fit(X, y)
        joblib.dump(stacking_pipeline, f'data_out/stacking_comparison/best_stacking_{n}_models.joblib')

# Save summary
results_df = pd.DataFrame(results, columns=['Num_Models', 'Model_Names', 'Mean_CV_Accuracy', 'Std_CV_Accuracy', 'CV_Scores'])
results_df.to_csv('data_out/stacking_comparison/stacking_comparison_results.csv', index=False)
print("\nStacking comparison completed. Results saved to data_out/stacking_comparison/stacking_comparison_results.csv") 