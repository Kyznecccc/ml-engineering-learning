import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, column_name: str, strategy: str = "mean"):
        self.column_name = column_name
        self.strategy = strategy
        self.fill_value = None  # Сюда сохраним среднее после fit()

    def fit(self, df: pd.DataFrame) -> None:
        if self.column_name not in df.columns:
            raise ValueError(f'Колонка {self.column_name} не найдена!')

        if self.strategy == "mean":
            self.fill_value = df[self.column_name].mean()

        elif self.strategy == 'median':
            self.fill_value = df[self.column_name].median()

        elif self.strategy == 'constant':
            self.fill_value = 0

        print(f"Fit completed. Fill value for {self.column_name}: {self.fill_value}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет заполнение пропусков к данным
        :param df: Датафрэйм с данными для трансформации
        :return: Датафрейм с заполнеными пропусками
        """
        if self.fill_value is None:
            raise ValueError(f"Сначала вызовите fit()! fill_value не установлен.")

        if self.column_name not in df.columns:
            raise ValueError(f"Колонка {self.column_name} не найдена!")

        df_copy = df.copy()

        df_copy[self.column_name] = df_copy[self.column_name].fillna(self.fill_value)

        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Выполняет fit и transform"""
        self.fit(df)
        return self.transform(df)

if __name__ == '__main__':
    # 1. Создаём тестовые данные с пропусками
    data = {
        'age': [25, 30, np.nan, 35, np.nan],
        'salary': [50000, 60000, 70000, np.nan, 90000]
    }
    df = pd.DataFrame(data)
    
    print("=== До обработки ===")
    print(df)
    print()
    
    # 2. Создаём препроцессор (указываем колонку!)
    preprocessor = DataPreprocessor(column_name="age", strategy="mean")
    
    # 3. Применяем fit_transform
    df_transformed = preprocessor.fit_transform(df)
    
    print("=== После обработки ===")
    print(df_transformed)
    print()
    
    # 4. Проверяем, что пропуски заполнены
    print(f"Заполненное значение: {preprocessor.fill_value}")
    print(f"Пропусков осталось: {df_transformed['age'].isna().sum()}")
