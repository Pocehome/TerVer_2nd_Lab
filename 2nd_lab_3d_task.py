import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import kagglehub

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None    # Коэффициенты (аналог model.coef_ в sklearn)
        self.intercept_ = None  # Свободный член (аналог model.intercept_)
        
    def fit(self, X, y):
        """Обучение модели (аналог model.fit(X, y))"""
        X = np.array(X)
        y = np.array(y)
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Вычисляем коэффициенты по методу наименьших квадратов
        try:
            theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        except np.linalg.LinAlgError:
            # Попробуем псевдообратную матрицу если матрица вырожденная
            theta = np.linalg.pinv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        
    def predict(self, X):
        """Предсказание (аналог model.predict(X))"""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Модель еще не обучена! Сначала вызовите fit().")
        return self.intercept_ + np.dot(X, self.coef_)
    
    def score(self, X, y):
        """R² score (аналог model.score(X, y))"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_total = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_total)


class MyPolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.linear_model = MyLinearRegression()
        
    def fit(self, X, y):
        """Обучение полиномиальной модели"""
        X_poly = self._transform_features(X)
        self.linear_model.fit(X_poly, y)
        
    def predict(self, X):
        """Предсказание полиномиальной модели"""
        X_poly = self._transform_features(X)
        return self.linear_model.predict(X_poly)
        
    def score(self, X, y):
        """R² score для полиномиальной модели"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_total = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_total)
        
    def _transform_features(self, X):
        """Преобразование признаков в полиномиальные"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Создаем матрицу для полиномиальных признаков
        X_poly = np.ones((n_samples, 1))  # Начинаем с единиц (для intercept)
        
        for d in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**d]
            
        return X_poly

# Загрузка данных
if not 'path' in locals():
    path = kagglehub.dataset_download("nadezhda2019/bikes-rent")
    print("Dataset downloaded to:", path)

df = pd.read_csv(os.path.join(path, "bikes_rent.csv"))
x_real = df["temp"].to_numpy().reshape(-1, 1) # Температура
y_real = df["cnt"].to_numpy() # Количество аренд

# Линейная регрессия
model_real_linear = MyLinearRegression()
model_real_linear.fit(x_real, y_real)
r2_real_linear = model_real_linear.score(x_real, y_real)
print(f"\nЛинейная регрессия: R^2 = {r2_real_linear:.4f}")
print(f"Коэффициенты: a = {model_real_linear.coef_[0]:.4f}, b = {model_real_linear.intercept_:.4f}\n")

# Визуализация линейной регрессии
plt.figure(figsize=(10, 6))
plt.scatter(x_real, y_real, alpha=0.5, label='Реальные данные')
x_range = np.linspace(x_real.min(), x_real.max(), 100).reshape(-1, 1)
y_pred_real_linear = model_real_linear.predict(x_range)
plt.plot(x_range, y_pred_real_linear, color='red', label='Линейная регрессия')
plt.xlabel("Нормализованная температура")
plt.ylabel("Количество аренд (cnt)")
plt.title("Линейная регрессия: Зависимость проката роликовых коньков от температуры")
plt.legend()
plt.grid(True)
plt.show()

# Полиномиальные регрессии разных степеней
degrees = [2, 3, 4, 5]
r2_scores = [r2_real_linear]

for degree in degrees:
    model_poly = MyPolynomialRegression(degree=degree)
    model_poly.fit(x_real, y_real)
    r2 = model_poly.score(x_real, y_real)
    r2_scores.append(r2)
    print(f"Полиномиальная регрессия (степень {degree}): R^2 = {r2:.4f}\n")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.scatter(x_real, y_real, alpha=0.5, label='Реальные данные')
    x_range = np.linspace(x_real.min(), x_real.max(), 100).reshape(-1, 1)
    y_pred_poly = model_poly.predict(x_range)
    plt.plot(x_range, y_pred_poly, color='green', label=f'Полиномиальная регрессия (степень {degree})')
    plt.xlabel("Нормализованная температура")
    plt.ylabel("Количество аренд (cnt)")
    plt.title(f"Полиномиальная регрессия (степень {degree}): Зависимость проката роликовых коньков от температуры")
    plt.legend()
    plt.grid(True)
    plt.show()

# Сравнение моделей
# print("\nСравнение моделей регрессии для реальных данных:")
# print(f"Линейная регрессия: R^2 = {r2_scores[0]:.4f}")
# for i, degree in enumerate(degrees, 1):
#     print(f"Полиномиальная регрессия (степень {degree}): R^2 = {r2_scores[i]:.4f}")