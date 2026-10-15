import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline as SklearnPipeline

import Data_preprocessing as dp
import Data_acquisition as da

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import joblib as jl


# ============================================================
# MODELS (Updated with Class Weights where applicable)
# ============================================================

models = [

    (
        "RandomForest",
        RandomForestClassifier(
            n_estimators=800,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",  # <-- Added class balancing
            random_state=42,
            n_jobs=-1
        )
    ),

    (
        "ExtraTrees",
        ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=False,
            criterion="entropy",
            class_weight="balanced",  # <-- Added class balancing
            random_state=42,
            n_jobs=-1
        )
    ),

    (
        "XGBoost",
        XGBClassifier(
            n_estimators=800,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=3,
            subsample=0.80,
            colsample_bytree=0.80,
            gamma=0.10,
            reg_alpha=0.10,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
    ),

    (
        "CatBoost",
        CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=5,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=5,
            random_strength=1.0,
            bagging_temperature=1.0,
            border_count=128,
            auto_class_weights="Balanced",  # <-- Added class balancing
            verbose=False,
            random_seed=42,
            thread_count=-1
        )
    ),

    (
        "LightGBM",
        LGBMClassifier(
            n_estimators=800,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=2.0,
            class_weight="balanced",  # <-- Added class balancing
            objective="binary",
            verbosity=-1,
            random_state=42,
            n_jobs=-1
        )
    ),

    (
        "HistGradientBoosting",
        HistGradientBoostingClassifier(
            max_iter=800,
            learning_rate=0.035,
            max_leaf_nodes=15,
            max_depth=None,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42
        )
    ),

    (
        "GradientBoosting",
        GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.80,
            max_features=None,
            random_state=42
        )
    ),

    (
        "AdaBoost",
        AdaBoostClassifier(
            n_estimators=500,
            learning_rate=0.03,
            random_state=42
        )
    ),

    (
        "LogisticRegression",
        SklearnPipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2000,
                        class_weight="balanced",  # <-- Added class balancing
                        random_state=42
                    )
                )
            ]
        )
    ),

    (
        "SVC",
        SklearnPipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",  # <-- Added class balancing
                        random_state=42
                    )
                )
            ]
        )
    )

]


# ============================================================
# PIPELINE
# ============================================================

class Pipeline:

    def __init__(
        self,
        data_target: pd.DataFrame,
        data_features: pd.DataFrame,
        model=None
    ):
        self.model = model if model is not None else RandomForestClassifier(
            n_estimators=800,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        self.data_target = data_target
        self.data_features = data_features
        
        # New variable to store the automatically tuned threshold
        self.optimal_threshold = 0.50

    def split_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Splits the target and features data into training and testing sets.
        The data is NOT shuffled because this is a time-series problem.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_features,
            self.data_target,
            test_size=test_size,
            random_state=random_state,
            shuffle=False
        )

        return X_train, X_test, y_train, y_test

    def find_best_threshold(self, X, y):
        """
        Scans thresholds between 0.30 and 0.70 on the provided data 
        and returns the one that maximizes the Matthews Correlation Coefficient.
        """
        if not hasattr(self.model, "predict_proba"):
            return 0.50

        probs = self.model.predict_proba(X)[:, 1]
        thresholds = np.linspace(0.3, 0.7, 41)
        
        best_thresh = 0.50
        best_mcc = -1.0
        
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            score = matthews_corrcoef(y, preds)
            
            if score > best_mcc:
                best_mcc = score
                best_thresh = thresh
                
        return best_thresh

    def train_model(self):
        """
        Trains the model using the training data and finds the best probability threshold.
        """
        X_train, X_test, y_train, y_test = self.split_data()

        self.model.fit(
            X_train,
            y_train
        )
        
        # Calculate optimal decision threshold based on training behavior
        self.optimal_threshold = self.find_best_threshold(X_train, y_train)

        return (
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def evaluate_model(self):
        """
        Evaluates the model using the testing data and the optimal threshold.
        """
        module, X_train, X_test, y_train, y_test = self.train_model()

        # ====================================================
        # PROBABILITY
        # ====================================================
        probability = None

        if hasattr(module, "predict_proba"):
            probability = module.predict_proba(X_test)[:, 1]

        # ====================================================
        # PREDICTION USING TUNED THRESHOLD
        # ====================================================
        if probability is not None:
            # Shift threshold automatically to correct majority class bias
            y_p = (probability >= self.optimal_threshold).astype(int)
        else:
            # Fallback for models without probability distributions
            y_p = module.predict(X_test)


        # ====================================================
        # EVALUATION
        # ====================================================
        all_evaluations = {
            "optimal_threshold": self.optimal_threshold,

            "classification_report":
                classification_report(
                    y_test,
                    y_p,
                    zero_division=0
                ),

            "confusion_matrix":
                confusion_matrix(
                    y_test,
                    y_p
                ),

            "accuracy":
                accuracy_score(
                    y_test,
                    y_p
                ),

            "f1_score":
                f1_score(
                    y_test,
                    y_p,
                    zero_division=0
                ),

            "precision":
                precision_score(
                    y_test,
                    y_p,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    y_p,
                    zero_division=0
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    y_test,
                    y_p
                ),

            "mcc":
                matthews_corrcoef(
                    y_test,
                    y_p
                )
        }

        # ====================================================
        # PROBABILITY METRICS
        # ====================================================
        if probability is not None:
            all_evaluations["roc_auc"] = roc_auc_score(
                y_test,
                probability
            )

            all_evaluations["pr_auc"] = average_precision_score(
                y_test,
                probability
            )

        self.save_model(
            module
        )

        return all_evaluations

    def save_model(
        self,
        model,
        filename: str = "trained_model.pkl"
    ):
        """
        Saves the trained model to a file.
        """
        jl.dump(
            model,
            filename
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # LOAD DATA
    # ========================================================
    df = pd.read_csv(
        "data_preprocessed.csv"
    )

    # ========================================================
    # SEPARATE TARGET AND FEATURES
    # ========================================================
    name = df.columns

    t = []
    f = []

    for i in name:
        if i.lower() != "Datetime".lower():
            if "lag" in i:
                f.append(i)
            else:
                t.append(i)

    t = df[t]
    f = df[f]

    print(t.head())
    print(f.head())

    # ========================================================
    # TARGET
    # ========================================================
    t = t["volatile"]

    # ========================================================
    # SHOW DATA INFORMATION
    # ========================================================
    print("\nTarget distribution:")
    print(t.value_counts())

    print("\nTarget proportion:")
    print(t.value_counts(normalize=True))

    print("\nFeature shape:")
    print(f.shape)

    print("\nTarget shape:")
    print(t.shape)


    # ========================================================
    # TRAIN ALL MODELS
    # ========================================================
    all_results = {}

    for name, model in models:
        print("\n" + "=" * 70)
        print("MODEL:", name)
        print("=" * 70)

        p = Pipeline(
            data_target=t,
            data_features=f,
            model=model
        )

        all = p.evaluate_model()

        all_results[name] = all


        # ====================================================
        # PRINT RESULTS
        # ====================================================
        
        print("\nOptimal Probability Threshold:", round(all["optimal_threshold"], 4))

        print("\nClassification Report:")
        print(all["classification_report"])

        print("Confusion Matrix:")
        print(all["confusion_matrix"])

        print("Accuracy:", all["accuracy"])
        print("F1 Score:", all["f1_score"])
        print("Precision:", all["precision"])
        print("Recall:", all["recall"])
        print("Balanced Accuracy:", all["balanced_accuracy"])
        print("MCC:", all["mcc"])

        if "roc_auc" in all:
            print("ROC-AUC:", all["roc_auc"])

        if "pr_auc" in all:
            print("PR-AUC:", all["pr_auc"])


        # ====================================================
        # SAVE MODEL WITH UNIQUE NAME
        # ====================================================
        filename = (
            name.lower()
            .replace(" ", "_")
            + "_model.pkl"
        )

        p.save_model(
            p.model,
            filename
        )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    comparison = []

    for name, result in all_results.items():

        comparison.append({
            "model": name,
            "optimal_threshold": round(result["optimal_threshold"], 3),
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1_score"],
            "balanced_accuracy": result["balanced_accuracy"],
            "mcc": result["mcc"],
            "roc_auc": result.get("roc_auc", np.nan),
            "pr_auc": result.get("pr_auc", np.nan)
        })


    comparison = pd.DataFrame(comparison)

    # ========================================================
    # SORT BY MCC INSTEAD OF F1 TO PRIORITIZE CLASS 0 RECOVERY
    # ========================================================
    comparison = comparison.sort_values(
        by="mcc",
        ascending=False
    )

    print(
        comparison.to_string(
            index=False
        )
    )