import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


TARGET = "Outcome"


class ModelTrainer:
    def __init__(self):
        self.models = {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                ))
            ]),
            "random_forest": RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42
            ),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(
                    C=2.0,
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=42
                ))
            ]),
            "xgboost": XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
        }

    def prepare_features(self, train_df, val_df):
        if TARGET not in train_df.columns:
            raise ValueError(f"Target column '{TARGET}' not found")

        feature_columns = [c for c in train_df.columns if c != TARGET]

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET]

        X_val = val_df[feature_columns]
        y_val = val_df[TARGET]

        logging.info(f"Features used: {feature_columns}")

        return X_train, X_val, y_train, y_val

    def evaluate_model(self, model, X_val, y_val, model_name):
        y_pred = model.predict(X_val)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_proba = None

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred)
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_val, y_proba)

        cm = confusion_matrix(y_val, y_pred)
        self._plot_confusion_matrix(cm, model_name)

        return metrics, cm

    def _plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix — {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"/tmp/confusion_matrix_{model_name}.png")
        plt.close()

    def train_models(self, train_df, val_df):
        X_train, X_val, y_train, y_val = self.prepare_features(train_df, val_df)

        results = {}

        for model_name, model in self.models.items():
            try:
                logging.info(f"Training model: {model_name}")

                with mlflow.start_run(run_name=model_name, nested=True):
                    model.fit(X_train, y_train)

                    metrics, cm = self.evaluate_model(
                        model, X_val, y_val, model_name
                    )

                    for k, v in metrics.items():
                        mlflow.log_metric(k, v)

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("features_count", X_train.shape[1])
                    mlflow.log_metric("train_samples", len(X_train))
                    mlflow.log_metric("val_samples", len(X_val))

                    mlflow.sklearn.log_model(model, model_name)
                    mlflow.log_artifact(
                        f"/tmp/confusion_matrix_{model_name}.png", "plots"
                    )

                    results[model_name] = {
                        "model": model,
                        "metrics": metrics,
                        "run_id": mlflow.active_run().info.run_id
                    }

            except Exception as e:
                logging.error(f"Error training {model_name}: {e}")

        if not results:
            raise RuntimeError("No models trained successfully")

        return results

    def find_best_model(self, results, metric="f1_score"):
        best_name = max(
            results,
            key=lambda k: results[k]["metrics"][metric]
        )
        return best_name, results[best_name]
