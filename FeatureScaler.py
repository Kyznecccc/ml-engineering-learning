import pandas as pd
import numpy as np

class FeatureScaler:
    def __init__(self, column_name:str):
        self.column_name = column_name
        self.min_value = None
        self.max_value = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Вычисляет минимумы и максимумы в столбце
        """
        if self.column_name not in df.columns:
            raise ValueError(f'Колонка {self.column_name} не найдена!')

        self.min_value = df[self.column_name].min()
        self.max_value = df[self.column_name].max()

        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Копируем Датафрэйм и применяем формулу
        """
        if self.min_value is None or self.max_value is None:
            raise ValueError(f"Сначала вызовите fit()! values не установлены.")

        df_copy = df.copy()

        df_copy[self.column_name] = (
            (df_copy[self.column_name] - self.min_value) /
            (self.max_value - self.min_value)
        )

        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Выполняет fit и transform"""
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает данные к исходному масштабу.
        """

        if self.min_value is None or self.max_value is None:
            raise ValueError("Сначала вызовите fit()!")

        df_copy = df.copy()

        # Обратная формула: X_original = X_scaled * (max - min) + min
        df_copy[self.column_name] = (
                df_copy[self.column_name] * (self.max_value - self.min_value) +
                self.min_value
        )

        return df_copy


if __name__ == '__main__':
    data = {
        'age': [20, 30, 40, 50, 60],
        'salary': [30000, 50000, 70000, 90000, 110000]
    }
    df = pd.DataFrame(data)

    print("=== До масштабирования ===")
    print(df)

    scaler = FeatureScaler(column_name="age")
    df_scaled = scaler.fit_transform(df)

    print("=== После масштабирования ===")
    print(df_scaled)

    # Проверка: значения должны быть от 0 до 1
    print(f"Min age: {df_scaled['age'].min()}")
    print(f"Max age: {df_scaled['age'].max()}")

    # Проверка inverse_transform
    df_restored = scaler.inverse_transform(df_scaled)
    print("=== После обратного преобразования ===")
    print(df_restored)