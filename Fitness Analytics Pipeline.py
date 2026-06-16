import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler


class FitnessDataPipeline:
    def __init__(self, df):
        self.raw_data = df
        self.X = df.drop('Сожжено_ккал', axis=1)
        self.y = df['Сожжено_ккал']

    def check_collinearity(self, threshold=0.98):
        corr_matrix = self.X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

        if to_drop:
            print(f"Автоматически найдены и удалены зависимые признаки: {to_drop}")
            self.X = self.X.drop(columns=to_drop)
        else:
            print("Линейно зависимых признаков не найдено.")

    def scale_features(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_scaled, columns=self.X.columns)

    def train_evaluate_models(self):
        model_lr = LinearRegression()
        model_ridge = Ridge(alpha=100.0)
        model_lasso = Lasso(alpha=30.0)

        model_lr.fit(self.X, self.y)
        model_ridge.fit(self.X, self.y)
        model_lasso.fit(self.X, self.y)

        weights_df = pd.DataFrame({
            'Признак': self.X.columns,
            'Без_штрафа': model_lr.coef_,
            'Ridge_(L2)': model_ridge.coef_,
            'Lasso_(L1)': model_lasso.coef_
        })
        print("\n--- Веса Моделей ---")
        print(weights_df.round(2))


if __name__ == "__main__":
    # Симуляция данных IoT-сенсоров
    np.random.seed(42)
    n_samples = 1000

    steps = np.random.normal(8000, 3000, n_samples).astype(int)
    avg_hr = np.random.normal(75, 12, n_samples).astype(int)
    workout_min = np.random.normal(30, 20, n_samples).astype(int)
    distance_km = steps * 0.000762
    battery_level = np.random.randint(1, 100, n_samples)
    calories = 1500 + (steps * 0.04) + (avg_hr * 2.5) + (workout_min * 6) + np.random.normal(0, 100, n_samples)

    df = pd.DataFrame({
        'Шаги_шт': steps,
        'Дистанция_км': distance_km,
        'Средний_пульс': avg_hr,
        'Тренировка_мин': workout_min,
        'Заряд_батареи_%': battery_level,
        'Сожжено_ккал': calories
    })

    # Запуск пайплайна
    pipeline = FitnessDataPipeline(df)
    pipeline.check_collinearity()
    pipeline.scale_features()
    pipeline.train_evaluate_models()