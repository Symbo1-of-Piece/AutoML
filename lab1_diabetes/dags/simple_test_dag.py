from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def test_task():
    """Простая тестовая задача"""
    logging.info("Тестовая задача выполнена успешно!")
    return "success"

def another_test_task():
    """Еще одна тестовая задача"""
    logging.info("Вторая тестовая задача выполнена!")
    return "success"

# Определяем DAG
with DAG(
    'simple_test_dag',
    default_args=default_args,
    description='Простой тестовый DAG',
    schedule_interval=None,  # Запуск только вручную
    catchup=False,
    tags=['test'],
) as dag:

    task1 = PythonOperator(
        task_id='test_task_1',
        python_callable=test_task,
    )

    task2 = PythonOperator(
        task_id='test_task_2',
        python_callable=another_test_task,
    )

    task1 >> task2