import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import catboost as cb
import warnings
import os
from titanic_model import engineer_features, drop_unused_features
import time
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
os.makedirs('data_out', exist_ok=True)
os.makedirs('data_out/model_comparison', exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the training data"""
    print("Loading training data...")
    df = pd.read_csv('data_in/train.csv')
    
    # Apply feature engineering
    df = engineer_features(df)
    
    # Separate features and target
    y = df['Transported']
    X = drop_unused_features(df, is_training=True)
    
    return X, y

def create_preprocessing_pipeline():
    """Create the preprocessing pipeline"""
    # Handle categorical variables
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 
                       'Infant', 'NoMoney', 'StarboardBC', 'Route']
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent']
    
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

    return preprocessor

def get_models():
    """Define all models to compare"""
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        # 'SVC': SVC(random_state=42, probability=True),
        'GaussianNB': GaussianNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBClassifier': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LGBMClassifier': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'CatBoostClassifier': cb.CatBoostClassifier(iterations=100, random_state=42, verbose=False, allow_writing_files=False)
    }
    return models

def compare_models(X, y, cv_folds=5):
    """Compare all models using cross-validation"""
    preprocessor = create_preprocessing_pipeline()
    models = get_models()
    
    results = []
    
    print("Starting model comparison...")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Calculate statistics
        mean_score = scores.mean()
        std_score = scores.std()
        
        results.append({
            'Model': name,
            'Mean_Accuracy': mean_score,
            'Std_Accuracy': std_score,
            'Min_Score': scores.min(),
            'Max_Score': scores.max(),
            'Training_Time': training_time,
            'CV_Scores': scores
        })
        
        print(f"  Mean Accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        print(f"  Training Time: {training_time:.2f} seconds")
        print(f"  CV Scores: {scores}")
    
    return pd.DataFrame(results)

def save_results(results_df):
    """Save results to file"""
    # Sort by mean accuracy
    results_df = results_df.sort_values('Mean_Accuracy', ascending=False)
    
    # Save detailed results
    with open('data_out/model_comparison/model_comparison_results.txt', 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"Model: {row['Model']}\n")
            f.write(f"Mean Accuracy: {row['Mean_Accuracy']:.4f}\n")
            f.write(f"Std Accuracy: {row['Std_Accuracy']:.4f}\n")
            f.write(f"Min Score: {row['Min_Score']:.4f}\n")
            f.write(f"Max Score: {row['Max_Score']:.4f}\n")
            f.write(f"Training Time: {row['Training_Time']:.2f} seconds\n")
            f.write(f"CV Scores: {row['CV_Scores']}\n")
            f.write("-" * 50 + "\n\n")
    
    # Save summary to CSV
    summary_df = results_df[['Model', 'Mean_Accuracy', 'Std_Accuracy', 'Min_Score', 'Max_Score', 'Training_Time']].copy()
    summary_df.to_csv('data_out/model_comparison/model_comparison_summary.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  - data_out/model_comparison/model_comparison_results.txt")
    print(f"  - data_out/model_comparison/model_comparison_summary.csv")

def create_visualization(results_df):
    """Create visualization of results"""
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Mean Accuracy Comparison
    results_sorted = results_df.sort_values('Mean_Accuracy', ascending=True)
    bars1 = ax1.barh(results_sorted['Model'], results_sorted['Mean_Accuracy'])
    ax1.set_xlabel('Mean Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    # 2. Training Time Comparison
    results_sorted_time = results_df.sort_values('Training_Time', ascending=True)
    bars2 = ax2.barh(results_sorted_time['Model'], results_sorted_time['Training_Time'])
    ax2.set_xlabel('Training Time (seconds)')
    ax2.set_title('Model Training Time Comparison')
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}s', ha='left', va='center')
    
    # 3. Accuracy vs Training Time Scatter
    ax3.scatter(results_df['Training_Time'], results_df['Mean_Accuracy'], s=100, alpha=0.7)
    for i, row in results_df.iterrows():
        ax3.annotate(row['Model'], (row['Training_Time'], row['Mean_Accuracy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax3.set_xlabel('Training Time (seconds)')
    ax3.set_ylabel('Mean Accuracy')
    ax3.set_title('Accuracy vs Training Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot of CV scores
    cv_data = []
    model_names = []
    for _, row in results_df.iterrows():
        cv_data.extend(row['CV_Scores'])
        model_names.extend([row['Model']] * len(row['CV_Scores']))
    
    cv_df = pd.DataFrame({'Model': model_names, 'CV_Score': cv_data})
    sns.boxplot(data=cv_df, x='CV_Score', y='Model', ax=ax4, order=results_df.sort_values('Mean_Accuracy', ascending=False)['Model'])
    ax4.set_xlabel('Cross-Validation Score')
    ax4.set_title('Distribution of CV Scores')
    
    plt.tight_layout()
    plt.savefig('data_out/model_comparison/model_comparison_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Visualization saved to: data_out/model_comparison/model_comparison_visualization.png")

def main():
    """Main function to run the model comparison"""
    print("Titanic Model Comparison")
    print("=" * 50)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    # Compare models
    results_df = compare_models(X, y)
    
    # Display top results
    print("\n" + "=" * 80)
    print("TOP 5 MODELS BY ACCURACY:")
    print("=" * 80)
    top_5 = results_df.nlargest(5, 'Mean_Accuracy')
    for _, row in top_5.iterrows():
        print(f"{row['Model']:<25} | Accuracy: {row['Mean_Accuracy']:.4f} | Time: {row['Training_Time']:.2f}s")
    
    # Save results
    save_results(results_df)
    
    # Create visualization
    create_visualization(results_df)
    
    print("\nModel comparison completed!")

if __name__ == "__main__":
    main() 