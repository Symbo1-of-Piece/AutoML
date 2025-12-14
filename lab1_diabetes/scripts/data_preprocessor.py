import pandas as pd
import numpy as np
import logging

def validate_diabetes_data(df, expected_columns=None):
    """Валидация данных Diabetes"""
    if expected_columns is None:
        expected_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]

    if df.empty:
        logging.error("DataFrame пустой")
        return False

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logging.warning(f"Отсутствуют колонки: {missing_cols}")
        return False

    logging.info(f"Валидация прошла успешно. Размер данных: {df.shape}")
    return True


def preprocess_diabetes_data(train_df, test_df):
    """Предобработка и feature engineering для датасета Diabetes"""

    if not validate_diabetes_data(train_df):
        raise ValueError("Ошибка валидации тренировочных данных")

    if not validate_diabetes_data(
        test_df,
        [c for c in train_df.columns if c != 'Outcome']
    ):
        raise ValueError("Ошибка валидации тестовых данных")

    combined = pd.concat([train_df, test_df], ignore_index=True)
    logging.info(f"Исходные данные - Train: {train_df.shape}, Test: {test_df.shape}")

    # === Обработка пропусков (нули = пропуски) ===
    zero_as_nan_cols = [
        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
    ]

    for col in zero_as_nan_cols:
        combined[col] = combined[col].replace(0, np.nan)
        median_value = combined[col].median()
        combined[col].fillna(median_value, inplace=True)
        logging.info(
            f"{col}: заполнены пропуски медианой = {median_value}"
        )

    # === Feature Engineering ===

    combined['BMI_Category'] = pd.cut(
        combined['BMI'],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    combined['Age_Group'] = pd.cut(
        combined['Age'],
        bins=[0, 30, 45, 60, 120],
        labels=[0, 1, 2, 3]
    ).astype(int)

    combined['Glucose_BMI'] = combined['Glucose'] * combined['BMI']
    combined['Insulin_Glucose'] = combined['Insulin'] / (combined['Glucose'] + 1)

    # === Разделяем обратно ===
    processed_train = combined.iloc[:len(train_df)].copy()
    processed_test = combined.iloc[len(train_df):].copy()

    if 'Outcome' in processed_test.columns:
        processed_test = processed_test.drop('Outcome', axis=1)

    logging.info(
        f"После обработки - Train: {processed_train.shape}, "
        f"Test: {processed_test.shape}"
    )

    return processed_train, processed_test