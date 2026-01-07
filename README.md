# Cholesterol and Genetic/Weight Relationship Prediction Model

Machine Learning model to predict cholesterol-related heart disease risk based on genetic and weight-related factors.
  
## Overview 
   
This project uses machine learning to analyze the relationship between cholesterol levels, genetic factors, and weight-related indicators to predict whether a patient has a **YES** (high risk) or **NO** (low risk) for cholesterol-related heart disease.

## Features
 
- **Multiple ML Models**: Compares Logistic Regression, Random Forest, and SVM classifiers
- **High Accuracy**: Achieves ~85% accuracy on test data
- **Genetic Factors**: Considers sex, thalassemia, and other genetic markers 
- **Weight Indicators**: Analyzes age, blood pressure, and related metrics 
- **Easy Prediction**: Simple command-line interface for single or batch predictions
- **Model Persistence**: Saves and loads trained models for reuse

## Dataset

The model is trained on the `dataset_2190_cholesterol.csv` containing 303 patient records with the following features:

- **age**: Age in years
- **sex**: Sex (1 = male, 0 = female)
- **cp**: Chest pain type (1-4)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (1-3)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)


### Training the Model

Train and compare all three models (Logistic Regression, Random Forest, SVM):

```bash
python cholesterol_model.py
```

This will:
- Train all three models
- Display performance metrics for each
- Save the best performing model as `best_cholesterol_model.pkl`

### Making Predictions

#### Interactive Single Prediction

```bash
python predict.py
```

#### Batch Prediction from CSV

```bash
python predict.py your_data.csv
```

This will create `your_data_predictions.csv` with predictions for all patients.


## Model Performance

The Random Forest model achieves the best performance:

- **Test Accuracy**: 85.0%
- **Precision**: 88.0%
- **Recall**: 78.6%
- **F1 Score**: 83.0%
- **Cross-Validation**: 82.7% (±7.8%)

## Example Output

```
Patient Profile:
  Age: 45, Sex: Male (1)
  Chest Pain Type: 3, Resting BP: 130 mm Hg
  Fasting Blood Sugar: Normal (0)
  Resting ECG: 2, Max Heart Rate: 150
  Exercise Angina: No (0), Oldpeak: 2.0
  Slope: 2, CA: 1, Thal: 7
  Cholesterol: 250 mg/dl

Prediction: YES - High Risk
Confidence: 64.00%
```

## Files

- `cholesterol_model.py`: Main ML model implementation
- `predict.py`: Simple prediction interface
- `dataset_2190_cholesterol.csv`: Training dataset
- `best_cholesterol_model.pkl`: Saved trained model
- `requirements.txt`: Python dependencies

## License

This project is licensed under the terms included in the LICENSE file.

## Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
