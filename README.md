# Predicting Diabetes 

> The project Predicting Diabetes is a comprehensive machine learning project for predicting diabetes stages using health indicators. This project implements an end-to-end pipeline that handles data preprocessing, feature engineering, model training, and evaluation using multiple algorithms. 

---

## Project Overview

This project presents a comprehensive machine learning pipeline for diabetes prediction using a dataset of 100,000 patient records. The study develops and compares two distinct classification approaches:

1. **Multi-Class Classification:** Distinguishing between 5 specific categories: No Diabetes, Pre-Diabetes, Type 1, Type 2, and Gestational Diabetes.
2. **Binary Classification:** A simplified approach distinguishing only between "Diabetes" and "No Diabetes".

The goal is to determine which approach offers more reliable predictions for initial screening purposes. The system implements advanced techniques including SMOTE for class imbalance, hybrid feature selection, and ensemble learning models (Random Forest, XGBoost, CatBoost).

### Key Objectives

- **Objective:** To determine which clinical health indicators and demographic factors are the greatest predictors of diabetes type and presence in patients.
- **Domain:** Healthcare
- **Key Techniques:** Exploratory Data Analysis, Feature Engineering, One-Hot Encoding, Multi-Class Classification, Binary Classification, SMOTE Resampling, Random Forest, XGBoost, CatBoost, Feature Selection (Mutual Information + Random Forest Importance).

---

## Project Structure

```
predicting-diabetes/
│
├── README.md                                         # Main project documentation
├── requirements.txt                                  # Python dependencies
│
├── data/
│   ├── diabetes_dataset.csv                          # Original dataset (100,000 records)
│   ├── prediction_results.csv                        # Model prediction results 
│   └── test_patients.csv                             # Input records used to generate predictions
│ 
├── code/
│   ├── Main.py                                       # Main pipeline implementation  
│   └── test_model.py                      
│
├── outputs/
│   ├── best_model_improved.pkl                       # Model OneDrive link mentioned below
│   ├── scaler_improved.pkl                           # StandardScaler object
│   └── label_encoder_improved.pkl                    # Label encoder object
│
│
├── report/
│   ├── Predicting_Diabetes_Project-Report.pdf        # Final comprehensive report
│ 
├── Presentation Slides/
│   ├── Predicting_Diabetes_Slide_Deck                # PowerPoint deck
│   
└── visualizations/
    ├── confusion_matrix_binary.png                   # Binary classification
    ├── confusion_matrix_multiclass.png               # Multi-class classification
    ├── cv_f1_scores_binary.png                       # Binary CV performance
    ├── cv_f1_scores_multi.png                        # Multi-class CV performance
    ├── figure1_feature_comparison.png                # Top 10 features
    ├── figure4_multi_vs_binary.png                   # Multi-class vs Binary
    ├── figure5_cv_vs_test.png                        # cv vs test gap
    ├── figure6_sensitivity_specificity.png           # Tradeoff curve
    ├── test_accuracy__binary.png                     # Binary model comparison
    └── test_accuracy_multi.png                       # Multi-class model comparison   
    
    

```
Best Model Imporved-Pkl file: [best_model_improved.pkl](https://redhawks-my.sharepoint.com/:u:/g/personal/smurala_seattleu_edu/IQAP5mDpUOeNSpwju-D3cHfYAUNH-AWlvl7IDpWNeIlO9Co?e=ddI6Tv)

## How to Run
1. Install dependencies listed in `requirements.txt`.
2. To run the main pipeline execute: `Main.py`.
3. To load the saved model and generate predictions on test data execute: `test_model.py`.
4. The existing model is saved in this location [best_model_improved.pkl](https://redhawks-my.sharepoint.com/:u:/g/personal/smurala_seattleu_edu/IQAP5mDpUOeNSpwju-D3cHfYAUNH-AWlvl7IDpWNeIlO9Co?e=ddI6Tv). It was trained on Python 3.9
---

## Data

- **Source:** Kaggle - [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset/data)

- **Size:** 14.37 MB

- **Records:** 100,000 patient records

- **Features:** 31 columns including demographic, lifestyle, and clinical health measurements

### Feature Categories:

- **Demographic Features:** Age, gender, ethnicity, education level, income level

- **Lifestyle Features:** Physical activity, diet score, alcohol consumption, smoking status, screen time (hours/day)

- **Clinical Measurements:** BMI, waist-to-hip ratio, blood pressure (systolic, diastolic), cholesterol levels (total, HDL, LDL), triglycerides, insulin levels

- **Medical History:** Hypertension history, family diabetes history

- **Target Variable:** Diabetes stage (5 classes: No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational)

- **License:** CC0: Public [Domain](https://creativecommons.org/publicdomain/zero/1.0/)

## Methodology

- **Feature Selection:** Voting consensus (Mutual Information + Random Forest Importance) → 10 selected features
  
- **Class Imbalance:** SMOTE resampling on training data
  
- **Models:** Random Forest, XGBoost, CatBoost with GridSearchCV (5-fold CV)
  
- **Scaling:** StandardScaler (z-score normalization)

---

## Analysis

The comprehensive analysis reveals that **problem formulation fundamentally constrains model performance**. While ensemble methods are powerful, distinguishing between five diabetes types using only demographic and lifestyle data proves inherently difficult due to:

1. **Clinical Overlap:** Type 2 and Pre-diabetes share similar presentations; Type 1 requires autoantibody testing
2. **Feature Insufficiency:** Excluding biomarkers (HbA1c, glucose) limits discriminability between types
3. **Severe Class Imbalance:** Type 1 (0.10%) and Gestational diabetes (0.22%) represent <1% of data

The binary classification framework provides a practical foundation for initial screening systems when integrated with confirmatory testing protocols.



## Results

### Binary Classification (Random Forest - Best Model)
| Metric | Value |
|--------|-------|
| Accuracy | 84.88% |
| F1-Score | 0.5342 |
| Sensitivity | 91% |
| Specificity | 17% |

### Multi-Class Classification (Random Forest)
| Metric | Value |
|--------|-------|
| Accuracy | 52.23% |
| F1-Score | 0.2239 |

### Top Features
1. Physical Activity
2. Age
3. BMI
4. Systolic Blood Pressure
5. Waist-to-Hip Ratio

## Key Findings

- Binary classification is more suitable for screening due to simpler decision boundary
- Random Forest provides best F1-macro score for balanced performance
- Lifestyle factors (physical activity, age) are strongest predictors
- Model suitable for preliminary screening with confirmatory testing required
  
## Authors

-  Sravya Murala - [Github Link](https://github.com/Sravyasss)
-  Jacob Wilson - [Github Link](https://github.com/jwilsonc)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

**Tools & Libraries Used**

- pandas & numpy: Data manipulation and numerical computing

- scikit-learn: Machine learning algorithms, metrics, and preprocessing

- XGBoost & CatBoost: Advanced gradient boosting implementations

- imbalanced-learn: SMOTE resampling for class imbalance handling

- matplotlib & seaborn: Data visualization


**References & Inspiration**

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
- World Health Organization. (2025). Diabetes. Retrieved from the [link](https://www.who.int/health-topics/diabetes)
- American Diabetes Association (2021). Classification and diagnosis of diabetes: Standards of medical care in diabetes—2021. 








