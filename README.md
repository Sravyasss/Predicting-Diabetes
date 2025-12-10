# Predicting Diabetes Progression

> The project Predicting Diabetes progression is a comprehensive machine learning project for predicting diabetes stages using health indicators. This project implements an end-to-end pipeline that handles data preprocessing, feature engineering, model training, and evaluation using multiple algorithms. 

---

## Project Overview

This project presents a comprehensive machine learning pipeline for diabetes prediction using a dataset of 100,000 patient records. The study develops and compares two distinct classification approaches:

1.  **Multi-Class Classification:** Distinguishing between 5 specific categories: No Diabetes, Pre-Diabetes, Type 1, Type 2, and Gestational Diabetes.
2.  **Binary Classification:** A simplified approach distinguishing only between "Diabetes" and "No Diabetes".

The goal is to determine which approach offers more reliable predictions for initial screening purposes. The system implements advanced techniques including SMOTE for class imbalance, hybrid feature selection, and ensemble learning model

- **Objective:** To determine which clinical health indicators and demographic factors are the greatest predictors of diabetes type and presence in patients.
- **Domain:** Healthcare
- **Key Techniques:** Exploratory Data Analysis, Feature Engineering, One-Hot Encoding, Multi-Class Classification, Binary Classification, SMOTE Resampling, Random Forest, XGBoost, CatBoost.

---

## Project Structure

```
├── data/
│   └── diabetes_dataset.csv          # 100,000 patient records with 31 features
├── code/
│   └── Main.py                       # Complete pipeline implementation
├── reports/
│   └── diabetes_report.pdf           # Comprehensive analysis report
├── models/
│   ├── best_model_improved.pkl       # Trained best performing model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── label_encoder.pkl             # Categorical label encode

```

---

## Data

- **Source:** Kaggle - Diabetes Health Indicators Dataset (https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset/data)

- **Size:** 14.37 MB

- **Records:** 100,000 patient records

- **Features:** 31 columns including demographic, lifestyle, and clinical health measurements

- **Description:** The dataset contains comprehensive health indicators for diabetes prediction, including:

- **Demographic Features:** Age, gender, ethnicity, education level, income level

- **Lifestyle Features:** Physical activity, diet score, alcohol consumption, smoking status, screen time (hours/day)

- **Clinical Measurements:** BMI, waist-to-hip ratio, blood pressure (systolic, diastolic), cholesterol levels (total, HDL, LDL), triglycerides, insulin levels.

- **Medical History:** Hypertension history, family diabetes history
  
- **Target Variable:** Diabetes stage (5 classes: No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational)
  
- **License:** CC0: Public Domain (https://creativecommons.org/publicdomain/zero/1.0/)
---

## Analysis


---

## Results


---

## Authors

-  Sravya Murala - (https://github.com/Sravyasss)
-  Jacob Wilson - (https://github.com/jwilsonc)

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

- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.

- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.

- Medical literature on diabetes risk factors and machine learning applications in healthcare

