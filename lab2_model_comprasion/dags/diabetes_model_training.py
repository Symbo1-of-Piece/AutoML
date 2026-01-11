from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging
import os
import sys
import json

import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

# scripts
sys.path.append('/opt/airflow/scripts/lab2')
from data_loader import download_diabetes_data, save_data_locally, load_data_from_local
from data_preprocessor import preprocess_diabetes_data


# ================= MLflow =================
def setup_mlflow():
    mlflow_uri = "http://mlflow:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("diabetes_model_comparison")
    return MlflowClient()


# ================= Config =================
TARGET = "Outcome"
PROCESSED_PATH = "/tmp/diabetes_processed"

MODELS = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        ))
    ]),
    "random_forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        ))
    ]),
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
    "xgboost": Pipeline([
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])
}


# ================= Tasks =================
def download_data():
    train_df, test_df = download_diabetes_data()
    save_data_locally(train_df, test_df)
    logging.info(f"Downloaded data: train={train_df.shape}, test={test_df.shape}")


def preprocess_data():
    train_df, test_df = load_data_from_local()
    processed_train, processed_test = preprocess_diabetes_data(train_df, test_df)
    save_data_locally(processed_train, processed_test, PROCESSED_PATH)
    logging.info(f"Preprocessed data: train={processed_train.shape}")


def train_and_compare_models():
    mlflow_client = setup_mlflow()

    df, _ = load_data_from_local(PROCESSED_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    class_balance = y.value_counts(normalize=True).to_dict()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for model_name, pipeline in MODELS.items():
        with mlflow.start_run(run_name=model_name):
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=skf,
                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                return_estimator=True
            )

            metrics = {
                "accuracy": np.mean(scores["test_accuracy"]),
                "precision": np.mean(scores["test_precision"]),
                "recall": np.mean(scores["test_recall"]),
                "f1_score": np.mean(scores["test_f1"]),
                "roc_auc": np.mean(scores["test_roc_auc"])
            }

            mlflow.log_params({"model": model_name})
            mlflow.log_metrics(metrics)
            mlflow.log_dict(class_balance, "class_balance.json")

            best_estimator = scores["estimator"][np.argmax(scores["test_f1"])]
            mlflow.sklearn.log_model(best_estimator, artifact_path=model_name)

            results[model_name] = {
                "metrics": metrics,
                "run_id": mlflow.active_run().info.run_id
            }

    best_model = max(results.items(), key=lambda x: x[1]["metrics"]["f1_score"])
    best_model_name = best_model[0]

    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_f1", best_model[1]["metrics"]["f1_score"])

    return best_model_name


def register_best_model(**context):
    best_model_name = context["ti"].xcom_pull(task_ids="train_and_compare_models")
    mlflow_client = setup_mlflow()

    experiment = mlflow.get_experiment_by_name("diabetes_model_comparison")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    run = runs[runs["tags.mlflow.runName"] == best_model_name].iloc[0]

    model_uri = f"runs:/{run.run_id}/{best_model_name}"
    model_name = "diabetes-prediction-model"

    try:
        mlflow_client.create_registered_model(model_name)
    except Exception:
        pass

    version = mlflow_client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run.run_id
    )

    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Production"
    )

    logging.info(f"Registered {model_name} v{version.version}")


def create_report():
    mlflow_client = setup_mlflow()
    experiment = mlflow.get_experiment_by_name("diabetes_model_comparison")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    report = runs[[
        "tags.mlflow.runName",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1_score",
        "metrics.roc_auc"
    ]].sort_values("metrics.f1_score", ascending=False)

    path = "/tmp/diabetes_model_report.csv"
    report.to_csv(path, index=False)

    with mlflow.start_run(run_name="final_report"):
        mlflow.log_artifact(path)

    logging.info("Final report generated")


# ================= DAG =================
default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="diabetes_model_comparison",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["diabetes", "mlflow", "model-comparison"],
) as dag:

    t1 = PythonOperator(task_id="download_data", python_callable=download_data)
    t2 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t3 = PythonOperator(task_id="train_and_compare_models", python_callable=train_and_compare_models)
    t4 = PythonOperator(task_id="register_best_model", python_callable=register_best_model, provide_context=True)
    t5 = PythonOperator(task_id="create_report", python_callable=create_report)

    t1 >> t2 >> t3 >> [t4, t5]
