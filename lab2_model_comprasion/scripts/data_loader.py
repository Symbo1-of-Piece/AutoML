import pandas as pd
import os
import logging
import numpy as np


def download_diabetes_data():
    """Загрузка датасета Diabetes с fallback-логикой (как в Titanic)"""
    try:
        # === PRIMARY SOURCE ===
        data_url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        logging.info("Пробуем загрузить датасет Diabetes из основного источника")

        df = pd.read_csv(data_url)

        if df.empty:
            raise ValueError("Основной источник вернул пустой датасет")

        logging.info("Датасет успешно загружен из основного источника")

        return split_train_test(df)

    except Exception as e:
        logging.error(f"Ошибка при загрузке основного датасета: {e}")
        logging.info("Пробуем fallback: создание mock-данных")

        try:
            train_df = create_mock_diabetes_data(is_train=True)
            test_df = create_mock_diabetes_data(is_train=False)
            logging.info("Mock-данные Diabetes успешно созданы")
            return train_df, test_df

        except Exception as mock_error:
            logging.error(f"Не удалось создать mock-данные: {mock_error}")
            raise


def split_train_test(df, test_ratio=0.2):
    """Единая логика разбиения"""
    required_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]

    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

    test_size = int(len(df) * test_ratio)
    test_df = df.tail(test_size).copy()
    train_df = df.head(len(df) - test_size).copy()

    if 'Outcome' in test_df.columns:
        test_df = test_df.drop('Outcome', axis=1)

    logging.info(
        f"Данные подготовлены. "
        f"Train shape: {train_df.shape}, Test shape: {test_df.shape}"
    )

    return train_df, test_df


def create_mock_diabetes_data(is_train=True, n_samples=100):
    """Mock-данные Diabetes (полный аналог mock Titanic)"""
    np.random.seed(42)

    data = {
        'Pregnancies': np.random.randint(0, 10, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(50, 200),
        'BloodPressure': np.random.normal(70, 10, n_samples).clip(40, 120),
        'SkinThickness': np.random.normal(25, 8, n_samples).clip(5, 60),
        'Insulin': np.random.normal(100, 40, n_samples).clip(10, 300),
        'BMI': np.random.normal(30, 6, n_samples).clip(15, 60),
        'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, n_samples),
        'Age': np.random.randint(21, 70, n_samples),
    }

    if is_train:
        # простая логика генерации таргета
        prob = (
            (data['Glucose'] > 140).astype(int) +
            (data['BMI'] > 30).astype(int)
        ) / 2
        data['Outcome'] = np.random.binomial(1, prob)

    df = pd.DataFrame(data)

    # добавляем "грязь" как в реальных данных
    for col in ['Glucose', 'BMI', 'Insulin']:
        idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        df.loc[idx, col] = 0

    return df


def save_data_locally(train_df, test_df, path="/tmp/diabetes_processed"):
    os.makedirs(path, exist_ok=True)
    train_df.to_csv(f"{path}/train.csv", index=False)
    test_df.to_csv(f"{path}/test.csv", index=False)


def load_data_from_local(path="/tmp/diabetes_processed"):
    train_df = pd.read_csv(f"{path}/train.csv")
    test_df = pd.read_csv(f"{path}/test.csv")
    return train_df, test_df
