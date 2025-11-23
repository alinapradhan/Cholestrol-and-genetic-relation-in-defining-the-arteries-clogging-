"""
Cholesterol and Genetic/Weight Relationship ML Model
Predicts cholesterol-related heart disease risk (YES/NO) based on genetic and weight factors
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')


class CholesterolPredictor:
    """
    ML Model to predict cholesterol-related heart disease risk.
    Uses genetic factors (sex, thal, ca) and weight-related factors (age, weight indicators)
    along with cholesterol levels to predict YES/NO for disease risk.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor with specified model type.
        
        Args:
            model_type (str): Type of model - 'logistic', 'random_forest', or 'svm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath='dataset_2190_cholesterol.csv'):
        """
        Load and preprocess the cholesterol dataset.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            tuple: X (features), y (target)
        """
        # Load data
        df = pd.read_csv(filepath)
        
        # Handle missing values (represented as '?')
        df = df.replace('?', np.nan)
        df = df.dropna()
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any remaining rows with NaN values
        df = df.dropna()
        
        # Convert multi-class target (0-4) to binary (0=No disease, 1-4=Yes disease)
        df['target'] = (df['num'] > 0).astype(int)
        
        # Select features focusing on genetic and weight-related factors with cholesterol
        # Genetic factors: sex, thal, ca
        # Weight/body factors: age, trestbps (blood pressure), cp (chest pain type)
        # Additional relevant: fbs (fasting blood sugar), restecg, thalach, exang, oldpeak, slope
        feature_columns = ['age', 'sex', 'cp', 'trestbps', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'chol']
        
        X = df[feature_columns]
        y = df['target']
        
        self.feature_names = feature_columns
        
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model with the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model based on type
        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=random_state, probability=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1_score': f1_score(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, 
                                                          target_names=['No Disease', 'Disease'])
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    def predict(self, X):
        """
        Predict disease risk for new data.
        
        Args:
            X: Feature matrix (must have same features as training data)
            
        Returns:
            numpy.ndarray: Predictions (0=No, 1=Yes)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict probability of disease risk.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy.ndarray: Probability predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_single(self, age, sex, cp, trestbps, fbs, restecg, thalach, 
                      exang, oldpeak, slope, ca, thal, chol):
        """
        Predict disease risk for a single patient.
        
        Args:
            age: Age in years
            sex: Sex (1=male, 0=female)
            cp: Chest pain type (1-4)
            trestbps: Resting blood pressure (mm Hg)
            fbs: Fasting blood sugar > 120 mg/dl (1=true, 0=false)
            restecg: Resting electrocardiographic results (0-2)
            thalach: Maximum heart rate achieved
            exang: Exercise induced angina (1=yes, 0=no)
            oldpeak: ST depression induced by exercise
            slope: Slope of peak exercise ST segment (1-3)
            ca: Number of major vessels colored by flourosopy (0-3)
            thal: Thalassemia (3=normal, 6=fixed defect, 7=reversable defect)
            chol: Serum cholesterol in mg/dl
            
        Returns:
            tuple: (prediction, probability)
        """
        X = pd.DataFrame([[age, sex, cp, trestbps, fbs, restecg, thalach, 
                          exang, oldpeak, slope, ca, thal, chol]], 
                        columns=self.feature_names)
        
        prediction = self.predict(X)[0]
        probability = self.predict_proba(X)[0]
        
        result = "YES - High Risk" if prediction == 1 else "NO - Low Risk"
        confidence = probability[prediction] * 100
        
        return result, confidence
    
    def save_model(self, filepath='cholesterol_model.pkl'):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='cholesterol_model.pkl'):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {filepath}")


def train_and_compare_models(filepath='dataset_2190_cholesterol.csv'):
    """
    Train multiple models and compare their performance.
    
    Args:
        filepath (str): Path to the dataset
        
    Returns:
        dict: Results for all models
    """
    print("=" * 70)
    print("CHOLESTEROL & GENETIC/WEIGHT RELATIONSHIP PREDICTION")
    print("=" * 70)
    
    results = {}
    
    for model_type in ['logistic', 'random_forest', 'svm']:
        print(f"\n{'=' * 70}")
        print(f"Training {model_type.upper().replace('_', ' ')} Model")
        print(f"{'=' * 70}")
        
        predictor = CholesterolPredictor(model_type=model_type)
        X, y = predictor.load_data(filepath)
        
        print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        metrics = predictor.train(X, y)
        results[model_type] = {
            'predictor': predictor,
            'metrics': metrics
        }
        
        print(f"\nTraining Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Cross-Validation: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
    
    # Find best model
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print(f"{'=' * 70}")
    
    best_model = None
    best_score = 0
    
    for model_type, result in results.items():
        score = result['metrics']['test_accuracy']
        print(f"{model_type.upper().replace('_', ' ')}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model_type
    
    print(f"\nBest Model: {best_model.upper().replace('_', ' ')} (Accuracy: {best_score:.4f})")
    
    # Save the best model
    best_predictor = results[best_model]['predictor']
    best_predictor.save_model('best_cholesterol_model.pkl')
    
    return results, best_predictor


if __name__ == "__main__":
    # Train and compare models
    results, best_model = train_and_compare_models()
    
    # Example prediction
    print(f"\n{'=' * 70}")
    print("EXAMPLE PREDICTION")
    print(f"{'=' * 70}")
    
    print("\nPatient Profile:")
    print("  Age: 45, Sex: Male (1)")
    print("  Chest Pain Type: 3, Resting BP: 130 mm Hg")
    print("  Fasting Blood Sugar: Normal (0)")
    print("  Resting ECG: 2, Max Heart Rate: 150")
    print("  Exercise Angina: No (0), Oldpeak: 2.0")
    print("  Slope: 2, CA: 1, Thal: 7")
    print("  Cholesterol: 250 mg/dl")
    
    result, confidence = best_model.predict_single(
        age=45, sex=1, cp=3, trestbps=130, fbs=0, restecg=2,
        thalach=150, exang=0, oldpeak=2.0, slope=2, ca=1, thal=7, chol=250
    )
    
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
