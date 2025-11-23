"""
Visualization script for model performance and feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cholesterol_model import CholesterolPredictor, train_and_compare_models
import warnings
warnings.filterwarnings('ignore')


def visualize_model_performance(results):
    """
    Create visualizations for model performance comparison
    
    Args:
        results: Dictionary containing model results
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # 1. Model Comparison Bar Chart
    plt.subplot(2, 3, 1)
    model_names = [name.replace('_', ' ').title() for name in results.keys()]
    accuracies = [results[model]['metrics']['test_accuracy'] for model in results.keys()]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.7)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim(0.7, 0.9)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Precision, Recall, F1-Score Comparison
    plt.subplot(2, 3, 2)
    metrics_data = []
    for model_name in results.keys():
        metrics = results[model_name]['metrics']
        metrics_data.append([
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ])
    
    x = np.arange(len(results))
    width = 0.25
    
    plt.bar(x - width, [m[0] for m in metrics_data], width, label='Precision', color='#3498db', alpha=0.7)
    plt.bar(x, [m[1] for m in metrics_data], width, label='Recall', color='#2ecc71', alpha=0.7)
    plt.bar(x + width, [m[2] for m in metrics_data], width, label='F1-Score', color='#e74c3c', alpha=0.7)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision, Recall, and F1-Score', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0.7, 0.95)
    
    # 3. Confusion Matrix for Best Model
    plt.subplot(2, 3, 3)
    best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['test_accuracy'])
    cm = results[best_model_name]['metrics']['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {best_model_name.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # 4. Cross-Validation Scores
    plt.subplot(2, 3, 4)
    cv_means = [results[model]['metrics']['cv_mean'] for model in results.keys()]
    cv_stds = [results[model]['metrics']['cv_std'] for model in results.keys()]
    
    plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5, color=colors, alpha=0.7)
    plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
    plt.ylabel('CV Score (mean ± std)', fontsize=12)
    plt.ylim(0.7, 0.9)
    
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        plt.text(i, mean, f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 5. Feature Importance (Random Forest only)
    plt.subplot(2, 3, 5)
    if 'random_forest' in results:
        rf_model = results['random_forest']['predictor'].model
        if hasattr(rf_model, 'feature_importances_'):
            feature_names = results['random_forest']['predictor'].feature_names
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            plt.barh(range(len(indices)), importances[indices], color='#2ecc71', alpha=0.7)
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance', fontsize=12)
            plt.title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
    
    # 6. Training vs Test Accuracy
    plt.subplot(2, 3, 6)
    train_acc = [results[model]['metrics']['train_accuracy'] for model in results.keys()]
    test_acc = [results[model]['metrics']['test_accuracy'] for model in results.keys()]
    
    x = np.arange(len(results))
    width = 0.35
    
    plt.bar(x - width/2, train_acc, width, label='Training', color='#3498db', alpha=0.7)
    plt.bar(x + width/2, test_acc, width, label='Testing', color='#e74c3c', alpha=0.7)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0.7, 1.05)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'model_performance.png'")
    plt.close()


def visualize_data_distribution():
    """
    Create visualizations for data distribution
    """
    # Load data
    df = pd.read_csv('dataset_2190_cholesterol.csv')
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    # Create target variable
    df['target'] = (df['num'] > 0).astype(int)
    df['target_label'] = df['target'].map({0: 'No Disease', 1: 'Disease'})
    
    # Set style
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Cholesterol Distribution by Target
    plt.subplot(2, 3, 1)
    for target, label in [(0, 'No Disease'), (1, 'Disease')]:
        data = df[df['target'] == target]['chol']
        plt.hist(data, bins=20, alpha=0.6, label=label)
    plt.xlabel('Cholesterol (mg/dl)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Cholesterol Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    
    # 2. Age Distribution by Target
    plt.subplot(2, 3, 2)
    sns.violinplot(x='target_label', y='age', data=df, palette=['#3498db', '#e74c3c'])
    plt.xlabel('Disease Status', fontsize=12)
    plt.ylabel('Age (years)', fontsize=12)
    plt.title('Age Distribution by Disease Status', fontsize=14, fontweight='bold')
    
    # 3. Sex Distribution
    plt.subplot(2, 3, 3)
    sex_disease = df.groupby(['sex', 'target_label']).size().unstack()
    sex_disease.plot(kind='bar', stacked=False, color=['#3498db', '#e74c3c'], ax=plt.gca())
    plt.xlabel('Sex (0=Female, 1=Male)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Sex Distribution by Disease Status', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.legend(title='Status')
    
    # 4. Blood Pressure vs Cholesterol
    plt.subplot(2, 3, 4)
    colors = df['target'].map({0: '#3498db', 1: '#e74c3c'})
    plt.scatter(df['trestbps'], df['chol'], c=colors, alpha=0.6, s=50)
    plt.xlabel('Resting Blood Pressure (mm Hg)', fontsize=12)
    plt.ylabel('Cholesterol (mg/dl)', fontsize=12)
    plt.title('Blood Pressure vs Cholesterol', fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='No Disease'),
                      Patch(facecolor='#e74c3c', label='Disease')]
    plt.legend(handles=legend_elements)
    
    # 5. Target Distribution
    plt.subplot(2, 3, 5)
    target_counts = df['target_label'].value_counts()
    colors_pie = ['#3498db', '#e74c3c']
    plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    plt.title('Disease Status Distribution', fontsize=14, fontweight='bold')
    
    # 6. Correlation Heatmap of Key Features
    plt.subplot(2, 3, 6)
    key_features = ['age', 'sex', 'chol', 'trestbps', 'thalach', 'target']
    corr_matrix = df[key_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'data_distribution.png'")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("CHOLESTEROL PREDICTION - VISUALIZATION")
    print("=" * 70)
    
    # Train models and get results
    print("\nTraining models...")
    results, best_model = train_and_compare_models()
    
    # Create visualizations
    print("\nCreating performance visualizations...")
    visualize_model_performance(results)
    
    print("\nCreating data distribution visualizations...")
    visualize_data_distribution()
    
    print("\n" + "=" * 70)
    print("All visualizations completed!")
    print("  - model_performance.png: Model comparison and metrics")
    print("  - data_distribution.png: Data analysis and distributions")
    print("=" * 70)
