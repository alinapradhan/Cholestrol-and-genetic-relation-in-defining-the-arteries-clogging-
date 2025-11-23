"""
Simple prediction script for cholesterol risk assessment
Usage: python predict.py
"""

from cholesterol_model import CholesterolPredictor
import pandas as pd


def predict_cholesterol_risk():
    """
    Interactive script to predict cholesterol-related disease risk
    """
    print("=" * 70)
    print("CHOLESTEROL RISK PREDICTION")
    print("Genetic and Weight Relationship Analysis")
    print("=" * 70)
    
    # Load the trained model
    print("\nLoading trained model...")
    predictor = CholesterolPredictor()
    try:
        predictor.load_model('best_cholesterol_model.pkl')
    except FileNotFoundError:
        print("Model not found! Training a new model...")
        from cholesterol_model import train_and_compare_models
        _, predictor = train_and_compare_models()
    
    print("\n" + "=" * 70)
    print("Enter patient information:")
    print("=" * 70)
    
    # Get user input
    try:
        age = int(input("\n1. Age (years): "))
        sex = int(input("2. Sex (1=Male, 0=Female): "))
        cp = int(input("3. Chest Pain Type (1-4): "))
        trestbps = int(input("4. Resting Blood Pressure (mm Hg): "))
        chol = int(input("5. Serum Cholesterol (mg/dl): "))
        fbs = int(input("6. Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No): "))
        restecg = int(input("7. Resting ECG Results (0=Normal, 1=Abnormality, 2=Hypertrophy): "))
        thalach = int(input("8. Maximum Heart Rate Achieved: "))
        exang = int(input("9. Exercise Induced Angina (1=Yes, 0=No): "))
        oldpeak = float(input("10. ST Depression (oldpeak): "))
        slope = int(input("11. Slope of Peak Exercise ST Segment (1-3): "))
        ca = int(input("12. Number of Major Vessels (0-3): "))
        thal = int(input("13. Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect): "))
        
        # Make prediction
        result, confidence = predictor.predict_single(
            age=age, sex=sex, cp=cp, trestbps=trestbps, fbs=fbs, restecg=restecg,
            thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, 
            ca=ca, thal=thal, chol=chol
        )
        
        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        print(f"\nCholesterol-Related Heart Disease Risk: {result}")
        print(f"Confidence Level: {confidence:.2f}%")
        
        if "YES" in result:
            print("\n⚠️  WARNING: Patient shows HIGH RISK for cholesterol-related disease")
            print("   Recommendation: Consult with a healthcare professional")
        else:
            print("\n✓  Patient shows LOW RISK for cholesterol-related disease")
            print("  Recommendation: Continue healthy lifestyle practices")
        
        print("=" * 70)
        
    except ValueError as e:
        print(f"\nError: Invalid input. Please enter numeric values only.")
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled by user.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")


def predict_from_file(filepath):
    """
    Predict cholesterol risk for patients in a CSV file
    
    Args:
        filepath (str): Path to CSV file with patient data
    """
    print("=" * 70)
    print("BATCH CHOLESTEROL RISK PREDICTION")
    print("=" * 70)
    
    # Load model
    print("\nLoading trained model...")
    predictor = CholesterolPredictor()
    try:
        predictor.load_model('best_cholesterol_model.pkl')
    except FileNotFoundError:
        print("Model not found! Training a new model...")
        from cholesterol_model import train_and_compare_models
        _, predictor = train_and_compare_models()
    
    # Load data
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Make predictions
    X = df[predictor.feature_names]
    predictions = predictor.predict(X)
    probabilities = predictor.predict_proba(X)
    
    # Add results to dataframe
    df['prediction'] = ['YES - High Risk' if p == 1 else 'NO - Low Risk' for p in predictions]
    df['confidence'] = [prob[pred] * 100 for prob, pred in zip(probabilities, predictions)]
    
    # Save results
    output_file = filepath.replace('.csv', '_predictions.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\nPredictions saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total patients: {len(df)}")
    print(f"  High Risk: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
    print(f"  Low Risk: {len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(predictions)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Batch prediction from file
        predict_from_file(sys.argv[1])
    else:
        # Interactive prediction
        predict_cholesterol_risk()
