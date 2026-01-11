from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import mlflow
import io
import logging
import sys

# Путь к scripts lab2 (diabetes)
sys.path.append('/opt/airflow/scripts/lab2')

try:
    from data_loader import download_diabetes_data, save_data_locally, load_data_from_local
    from data_preprocessor import preprocess_diabetes_data
except ImportError as e:
    logging.error(f"Ошибка импорта скриптов: {e}")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "data_loader", "/opt/airflow/scripts/lab2/data_loader.py"
    )
    data_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader)
    download_diabetes_data = data_loader.download_diabetes_data
    save_data_locally = data_loader.save_data_locally
    load_data_from_local = data_loader.load_data_from_local

    spec = importlib.util.spec_from_file_location(
        "data_preprocessor", "/opt/airflow/scripts/lab2/data_preprocessor.py"
    )
    data_preprocessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_preprocessor)
    preprocess_diabetes_data = data_preprocessor.preprocess_diabetes_data


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def download_and_log_data():
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("diabetes_basic_pipeline")
        mlflow_enabled = True
    except Exception:
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("diabetes_basic_pipeline")
        mlflow_enabled = True

    train_df, test_df = download_diabetes_data()

    if mlflow_enabled:
        with mlflow.start_run(run_name="data_download"):
            mlflow.log_param("dataset", "diabetes")
            mlflow.log_metric("train_samples", len(train_df))
            mlflow.log_metric("test_samples", len(test_df))
            mlflow.log_metric("total_features", train_df.shape[1])

            buf = io.StringIO()
            train_df.info(buf=buf)
            logging.info(buf.getvalue())

    save_data_locally(train_df, test_df)


def preprocess_and_log_data():
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("diabetes_basic_pipeline")
        mlflow_enabled = True
    except Exception:
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("diabetes_basic_pipeline")
        mlflow_enabled = True

    train_df, test_df = load_data_from_local()
    processed_train, processed_test = preprocess_diabetes_data(train_df, test_df)

    if mlflow_enabled:
        with mlflow.start_run(run_name="data_preprocessing"):
            mlflow.log_param("missing_value_strategy", "median")
            mlflow.log_metric("processed_train_samples", len(processed_train))
            mlflow.log_metric("processed_features", processed_train.shape[1])

            logging.info(processed_train.head(10).to_string())

    save_data_locally(
        processed_train,
        processed_test,
        "/tmp/diabetes_processed_final"
    )


def log_dataset_summary():
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("diabetes_basic_pipeline")
        mlflow_enabled = True
    except Exception:
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("diabetes_basic_pipeline")
        mlflow_enabled = True

    train_df, test_df = load_data_from_local("/tmp/diabetes_processed_final")

    summary = {
        "total_samples": len(train_df) + len(test_df),
        "features_count": train_df.shape[1],
    }

    if mlflow_enabled:
        with mlflow.start_run(run_name="dataset_summary"):
            for k, v in summary.items():
                mlflow.log_metric(k, v)
            mlflow.log_param("features", str(list(train_df.columns)))

    logging.info(summary)


with DAG(
    dag_id='diabetes_basic_pipeline',
    default_args=default_args,
    description='Базовый пайплайн для датасета Diabetes',
    schedule_interval=None,
    catchup=False,
    tags=['diabetes', 'mlflow'],
) as dag:

    download_task = PythonOperator(
        task_id='download_diabetes_data',
        python_callable=download_and_log_data,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_diabetes_data',
        python_callable=preprocess_and_log_data,
    )

    summary_task = PythonOperator(
        task_id='log_dataset_summary',
        python_callable=log_dataset_summary,
    )

    download_task >> preprocess_task >> summary_task
