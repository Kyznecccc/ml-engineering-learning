import numpy as np

class IQRFilter:
    def __init__(self):
        # Здесь мы заводим переменные для хранения границ.
        # Пока мы ничего не посчитали, они равны None.
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, data):
        """
        Метод для 'обучения'. Он считает границы и ЗАПОМИНАЕТ их.
        """
        # 1. Считает 1-й квартиль
        Q1 = np.quantile(data, 0.25)
        # 2. Считает 3-й квартиль
        Q3 = np.quantile(data, 0.75)
        # 3. считает IQR
        IQR = Q3 - Q1
        # 4. Вычисляем нижнюю и верхнюю границы.
        self.lower_bound = Q1 - 1.5 * IQR
        self.upper_bound = Q3 + 1.5 * IQR


    def transform(self, data):
        """
        Метод для применения фильтра к любым данным.
        """
        # 1. Проверка от "дурака":
        if self.lower_bound is None or self.upper_bound is None:
            raise ValueError('Сначала вызовите метод fit!')

        # 2. Фильтруем и выводим список data.
        return [x for x in data if x >= self.lower_bound and x <= self.upper_bound]


train_data = [10, 12, 15, 14, 11, 13, 100] # 100 - это сильный выброс
test_data = [11, 14, -50, 12] # -50 - это аномалия в новых данных

cleaner = IQRFilter()

# 1. Алгоритм изучает данные и запоминает границы
cleaner.fit(train_data)

# 2. Применяем фильтр к тренировочным данным
print("Очищенный train:", cleaner.transform(train_data))
# Ожидается: [10, 12, 15, 14, 11, 13]

# 3. Применяем ТОТ ЖЕ фильтр к абсолютно новым данным
print("Очищенный test:", cleaner.transform(test_data))
# Ожидается: [11, 14, 12] (число -50 отфильтровано по правилам train_data)