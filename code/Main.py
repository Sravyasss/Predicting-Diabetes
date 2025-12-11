import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')


class DiabetesPrediction:

    def __init__(self, data_path, n_features=10, binary_classification=False):
        self.data_path = data_path
        self.n_features = n_features
        self.binary_classification = binary_classification
        self.best_model = None
        self.best_model_name = None
        self.selected_features = None

    def load_data(self):
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")

    def prepare_data(self):
        print("\n" + "=" * 80)
        print("PREPARING DATA")
        print("=" * 80)

        target_col = None
        for col in ['diabetes_stage', 'Diabetes_Stage', 'diabetes', 'Diabetes']:
            if col in self.df.columns:
                target_col = col
                break

        if not target_col:
            print("ERROR: Could not find target column!")
            return False

        if self.df[target_col].dtype == 'object':
            self.label_encoder = LabelEncoder()
            self.df['target'] = self.label_encoder.fit_transform(self.df[target_col])
        else:
            self.df['target'] = self.df[target_col]
            self.label_encoder = None

        if self.binary_classification:
            print("\nConverting to BINARY classification (Diabetes vs No Diabetes)")
            if self.label_encoder:
                no_diabetes_idx = [i for i, name in enumerate(self.label_encoder.classes_)
                                   if 'no' in name.lower() or 'normal' in name.lower()]
                no_diabetes_class = no_diabetes_idx[0] if no_diabetes_idx else 0
            else:
                no_diabetes_class = 0

            self.df['target'] = (self.df['target'] != no_diabetes_class).astype(int)
            print("Binary classes: 0 = No Diabetes, 1 = Diabetes")
            print("\nClass distribution:")
            print(self.df['target'].value_counts())

        exclude_keywords = ['diabetes', 'target', 'diagnosed']
        feature_cols = [c for c in self.df.columns
                        if c not in [target_col, 'target', 'target_binary']
                        and not any(kw in c.lower() for kw in exclude_keywords)]

        self.X_full = self.df[feature_cols].copy()
        self.y = self.df['target'].copy()

        self.original_features = self.X_full.copy()
        self.categorical_cols = self.X_full.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.X_full.select_dtypes(include=[np.number]).columns.tolist()

        if self.categorical_cols:
            self.X_full = pd.get_dummies(self.X_full, columns=self.categorical_cols, drop_first=True)

        print(f"\nFeature count after encoding: {self.X_full.shape[1]}")
        return True

    def select_best_features(self):
        print("\n" + "=" * 80)
        print(f"FEATURE SELECTION (Top {self.n_features})")
        print("=" * 80)

        exclude_features = [
            'diagnosed_diabetes', 'hba1c', 'glucose_postprandial',
            'glucose_fasting', 'diabetes_risk_score'
        ]

        available_features = [col for col in self.X_full.columns
                              if not any(exc in col.lower() for exc in exclude_features)]
        X_filtered = self.X_full[available_features].copy()

        print(f"\nUsing 2 feature selection methods:")
        print("1. Mutual Information")
        print("2. Random Forest Importance")

        mi_scores = mutual_info_classif(X_filtered, self.y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X_filtered.columns,
            'score': mi_scores
        }).sort_values('score', ascending=False)

        top_mi = set(mi_df.head(15)['feature'])

        rf_temp = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_temp.fit(X_filtered, self.y)

        importance_df = pd.DataFrame({
            'feature': X_filtered.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)

        top_rf = set(importance_df.head(15)['feature'])

        feature_votes = {}
        for feature in available_features:
            votes = 0
            if feature in top_mi:
                votes += 1
            if feature in top_rf:
                votes += 1
            feature_votes[feature] = votes

        sorted_features = sorted(
            feature_votes.items(),
            key=lambda x: (
                x[1],
                importance_df[importance_df['feature'] == x[0]]['importance'].values[0]
            ),
            reverse=True
        )

        self.selected_features = [f[0] for f in sorted_features[:self.n_features]]

        print(f"\nSelected {len(self.selected_features)} features:")
        for i, feature in enumerate(self.selected_features, 1):
            votes = feature_votes[feature]
            mi_score = mi_df[mi_df['feature'] == feature]['score'].values[0]
            rf_score = importance_df[importance_df['feature'] == feature]['importance'].values[0]
            print(f"{i}. {feature:40s} (Votes: {votes}/2, MI: {mi_score:.4f}, RF: {rf_score:.4f})")

        self.X = self.X_full[self.selected_features].copy()


        merged = pd.DataFrame({
            'feature': self.selected_features,
            'MI_Score': [mi_df[mi_df['feature'] == f]['score'].values[0] for f in self.selected_features],
            'RF_Importance': [importance_df[importance_df['feature'] == f]['importance'].values[0]
                              for f in self.selected_features]
        })

        merged.set_index('feature').plot(kind='bar', figsize=(12, 6))
        plt.title("Top 10 Selected Features Comparison\n(Mutual Information vs Random Forest Importance)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig("figure1_feature_comparison.png", dpi=300)
        plt.close()

    def split_data(self):
        print("\n" + "=" * 80)
        print("SPLITTING DATA")
        print("=" * 80)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")

    def balance_data(self):
        print("\n" + "=" * 80)
        print("BALANCING CLASSES (SMOTE)")
        print("=" * 80)

        print("\nClass distribution before SMOTE:")
        print(pd.Series(self.y_train).value_counts().sort_index())

        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        print("\nClass distribution after SMOTE:")
        print(pd.Series(self.y_train).value_counts().sort_index())

    def train_with_cross_validation(self):
        print("\n" + "=" * 80)
        print("TRAINING WITH CROSS-VALIDATION")
        print("=" * 80)

        self.models = {}
        self.cv_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        print("\n1. Random Forest with Grid Search")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_params,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        rf_grid.fit(self.X_train, self.y_train)

        self.models['Random Forest'] = rf_grid.best_estimator_
        self.cv_scores['Random Forest'] = rf_grid.best_score_
        print(f"Best parameters: {rf_grid.best_params_}")
        print(f"CV F1-Score: {rf_grid.best_score_:.4f}")

        print("\n2. XGBoost with Grid Search")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }

        xgb_grid = GridSearchCV(
            XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
            xgb_params,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        xgb_grid.fit(self.X_train, self.y_train)

        self.models['XGBoost'] = xgb_grid.best_estimator_
        self.cv_scores['XGBoost'] = xgb_grid.best_score_
        print(f"Best parameters: {xgb_grid.best_params_}")
        print(f"CV F1-Score: {xgb_grid.best_score_:.4f}")

        print("\n3. CatBoost")
        catboost_model = CatBoostClassifier(
            iterations=200,
            depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        catboost_model.fit(self.X_train, self.y_train)

        scores = cross_val_score(catboost_model, self.X_train, self.y_train,
                                 cv=cv, scoring='f1_macro', n_jobs=-1)

        self.models['CatBoost'] = catboost_model
        self.cv_scores['CatBoost'] = scores.mean()
        print(f"CV F1-Score: {scores.mean():.4f}")

        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SCORES SUMMARY")
        print("=" * 80)
        sorted_scores = sorted(self.cv_scores.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_scores:
            print(f"{name:20s}: {score:.4f}")

    def evaluate_models(self):
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 80)

        results = []

        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1_Macro': f1_macro,
                'F1_Weighted': f1_weighted
            })

        self.results_df = pd.DataFrame(results).sort_values('F1_Macro', ascending=False)

        print("\n" + self.results_df.to_string(index=False))

        best = self.results_df.iloc[0]
        self.best_model_name = best['Model']
        self.best_model = self.models[self.best_model_name]

        print(f"\nBest Model: {self.best_model_name}")
        print(f"Test F1-Score (Macro): {best['F1_Macro']:.4f}")
        print(f"Test Accuracy: {best['Accuracy'] * 100:.2f}%")

        return self.best_model_name, self.best_model

    def show_detailed_results(self):
        print("\n" + "=" * 80)
        print(f"DETAILED RESULTS - {self.best_model_name}")
        print("=" * 80)

        y_pred = self.best_model.predict(self.X_test)

        if self.binary_classification:
            target_names = ['No Diabetes', 'Diabetes']
        elif self.label_encoder:
            target_names = self.label_encoder.classes_
        else:
            target_names = [str(i) for i in sorted(self.y.unique())]

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)

        title = f'Confusion Matrix - {self.best_model_name}\n'
        title += f'({"Binary" if self.binary_classification else "Multi-class"} Classification, '
        title += f'{len(self.selected_features)} Features)'

        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # Save confusion matrix
        if self.binary_classification:
            plt.savefig('figure3_confusion_matrix_binary.png', dpi=300)
        else:
            plt.savefig('figure2_confusion_matrix_multi.png', dpi=300)

        plt.close()


        if self.binary_classification:
            y_prob = self.best_model.predict_proba(self.X_test)[:, 1]

            thresholds = np.linspace(0, 1, 200)
            sensitivity = []
            specificity = []

            for t in thresholds:
                y_hat = (y_prob >= t).astype(int)
                cm = confusion_matrix(self.y_test, y_hat)
                tn, fp, fn, tp = cm.ravel()

                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0

                sensitivity.append(sens)
                specificity.append(spec)

            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, sensitivity, label="Sensitivity")
            plt.plot(thresholds, specificity, label="Specificity")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title("Sensitivity–Specificity Trade-off")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("figure6_sensitivity_specificity.png", dpi=300)
            plt.close()

    def save_model(self):
        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)

        joblib.dump(self.best_model, 'best_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')

        with open('selected_features.txt', 'w') as f:
            f.write('\n'.join(self.selected_features))

        import json
        metadata = {
            'binary_classification': self.binary_classification,
            'n_features': self.n_features,
            'best_model': self.best_model_name,
            'test_f1_score': float(self.results_df.iloc[0]['F1_Macro']),
            'test_accuracy': float(self.results_df.iloc[0]['Accuracy'])
        }

        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Saved model files successfully")

    def run_complete_pipeline(self):
        print("\n" + "=" * 80)
        print("DIABETES PREDICTION PIPELINE")
        print("=" * 80)

        classification_type = "BINARY" if self.binary_classification else "MULTI-CLASS"
        print(f"\nClassification Type: {classification_type}")
        print(f"Features to Select: {self.n_features}")

        self.load_data()

        if not self.prepare_data():
            return False

        self.select_best_features()
        self.split_data()
        self.balance_data()
        self.train_with_cross_validation()
        self.evaluate_models()
        self.show_detailed_results()
        self.save_model()

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED")
        print("=" * 80)

        return True


if __name__ == "__main__":

    print("\n" + "#" * 80)
    print("# MULTI-CLASS CLASSIFICATION")
    print("#" * 80)

    pipeline_multi = DiabetesPrediction(
        "diabetes_dataset.csv",
        n_features=10,
        binary_classification=False
    )
    success = pipeline_multi.run_complete_pipeline()

    print("\n\n" + "#" * 80)
    print("# BINARY CLASSIFICATION")
    print("#" * 80)

    pipeline_binary = DiabetesPrediction(
        "diabetes_dataset.csv",
        n_features=10,
        binary_classification=True
    )
    success_binary = pipeline_binary.run_complete_pipeline()

    if success and success_binary:
        print("\n\n" + "=" * 80)
        print("COMPARISON: MULTI-CLASS vs BINARY")
        print("=" * 80)

        print(f"\nMulti-class (5 classes):")
        print(f"  F1-Score: {pipeline_multi.results_df.iloc[0]['F1_Macro']:.4f}")
        print(f"  Accuracy: {pipeline_multi.results_df.iloc[0]['Accuracy'] * 100:.2f}%")

        print(f"\nBinary (2 classes):")
        print(f"  F1-Score: {pipeline_binary.results_df.iloc[0]['F1_Macro']:.4f}")
        print(f"  Accuracy: {pipeline_binary.results_df.iloc[0]['Accuracy'] * 100:.2f}%")

        labels = ["Multi-class", "Binary"]
        f1_scores = [
            pipeline_multi.results_df.iloc[0]['F1_Macro'],
            pipeline_binary.results_df.iloc[0]['F1_Macro']
        ]
        accuracies = [
            pipeline_multi.results_df.iloc[0]['Accuracy'],
            pipeline_binary.results_df.iloc[0]['Accuracy']
        ]

        plt.figure(figsize=(8, 6))
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, f1_scores, width, label="F1 Macro")
        plt.bar(x + width/2, accuracies, width, label="Accuracy")

        plt.xticks(x, labels)
        plt.title("Model Performance Comparison: Multi-Class vs Binary")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figure4_multi_vs_binary.png", dpi=300)
        plt.close()


        plt.figure(figsize=(10, 6))

        models = pipeline_multi.cv_scores.keys()
        cv_scores = list(pipeline_multi.cv_scores.values())
        test_scores = [
            pipeline_multi.results_df.set_index("Model").loc[m]['F1_Macro']
            for m in models
        ]

        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width/2, cv_scores, width, label="Cross-validation F1")
        plt.bar(x + width/2, test_scores, width, label="Test F1")

        plt.xticks(x, models, rotation=45)
        plt.title("Cross-Validation vs Test Performance Gap (Multi-class)")
        plt.ylabel("F1 Macro Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figure5_cv_vs_test.png", dpi=300)
        plt.close()
