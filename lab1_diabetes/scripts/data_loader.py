import pandas as pd
import os
import logging

def download_diabetes_data():
    """Загрузка датасета Pima Indians Diabetes по прямой ссылке"""
    try:
        data_url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

        try:
            df = pd.read_csv(data_url)
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных с URL {data_url}: {e}")
            raise

        if df.empty:
            raise ValueError("Загружен пустой датасет")

        required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

        test_size = int(len(df) * 0.2)
        test_df = df.tail(test_size).copy()
        train_df = df.head(len(df) - test_size).copy()

        if 'Outcome' in test_df.columns:
            test_df = test_df.drop('Outcome', axis=1)

        logging.info(
            f"Данные загружены. "
            f"Train shape: {train_df.shape}, Test shape: {test_df.shape}"
        )

        return train_df, test_df

    except Exception as e:
        logging.error(f"Ошибка при загрузке данных Diabetes: {e}")
        raise


def save_data_locally(train_df, test_df, path="/tmp/diabetes_processed"):
    os.makedirs(path, exist_ok=True)
    train_df.to_csv(f"{path}/train.csv", index=False)
    test_df.to_csv(f"{path}/test.csv", index=False)


def load_data_from_local(path="/tmp/diabetes_processed"):
    train_df = pd.read_csv(f"{path}/train.csv")
    test_df = pd.read_csv(f"{path}/test.csv")
    return train_df, test_df
