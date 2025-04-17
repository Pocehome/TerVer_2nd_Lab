import numpy as np
import matplotlib.pyplot as plt

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
        theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        
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
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1 - (ss_res / ss_total)

def generate_sample(a, b, sigma, n):
    """Генерация выборки с шумом"""
    x = np.arange(1, n+1)
    y_true = a * x + b
    y_noisy = y_true + np.random.normal(0, sigma, n)
    return x.reshape(-1, 1), y_noisy

def generate_random_sample(a, b, sigma, n, t1, t2):
    """Генерация случайной выборки на отрезке [t1, t2]"""
    x = np.random.uniform(t1, t2, n)
    y_true = a * x + b
    y_noisy = y_true + np.random.normal(0, sigma, n)
    return x.reshape(-1, 1), y_noisy

def plot_results(x, y, x_new, y_new, y_pred, title):
    """Визуализация результатов"""
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Исходная выборка', alpha=0.7)
    plt.scatter(x_new, y_new, label='Дополнительная выборка', alpha=0.7)
    x_combined = np.concatenate([x, x_new])
    plt.plot(x_combined, np.concatenate([y_pred[:len(x)], y_pred[len(x):]]), 
             'r-', label='Предсказание модели')
    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def linear_regression_task():
    # Часть 1: x = (1, 2, ..., n)
    print("=== Часть 1: x = (1, 2, ..., n) ===")
    
    # Ввод параметров
    a = float(input("Введите коэффициент a: "))
    b = float(input("Введите коэффициент b: "))
    sigma = float(input("Введите стандартное отклонение sigma: "))
    n = int(input("Введите размер выборки n: "))
    
    # 1) Генерация выборки
    x, y = generate_sample(a, b, sigma, n)
    
    # 2) Обучение модели
    model = MyLinearRegression()
    model.fit(x, y)
    print(f"Оценки коэффициентов: a* = {model.coef_[0]:.3f}, b* = {model.intercept_:.3f}")
    
    # 3) Вычисление R²
    r2 = model.score(x, y)
    print(f"Коэффициент детерминации R²: {r2:.3f}")
    
    # 4) Генерация дополнительной выборки
    m = int(input("Введите размер дополнительной выборки m: "))
    x_new = np.arange(n+1, n+m+1).reshape(-1, 1)
    y_new = a * x_new.flatten() + b + np.random.normal(0, sigma, m)
    y_pred = model.predict(np.concatenate([x, x_new]))
    
    # Визуализация
    plot_results(x, y, x_new, y_new, y_pred, 
                "Линейная регрессия для x = (1, 2, ..., n)")
    
    # Часть 2: случайные x на отрезке [t1, t2]
    print("\n=== Часть 2: случайные x на отрезке [t1, t2] ===")
    t1 = float(input("Введите t1: "))
    t2 = float(input("Введите t2: "))
    
    # 1) Генерация случайной выборки
    x_rand, y_rand = generate_random_sample(a, b, sigma, n, t1, t2)
    
    # 2) Обучение модели
    model_rand = MyLinearRegression()
    model_rand.fit(x_rand, y_rand)
    print(f"Оценки коэффициентов: a* = {model_rand.coef_[0]:.3f}, b* = {model_rand.intercept_:.3f}")
    
    # 3) Вычисление R²
    r2_rand = model_rand.score(x_rand, y_rand)
    print(f"Коэффициент детерминации R²: {r2_rand:.3f}")
    
    # 4) Генерация дополнительной случайной выборки
    x_new_rand = np.random.uniform(t1, t2, m).reshape(-1, 1)
    y_new_rand = a * x_new_rand.flatten() + b + np.random.normal(0, sigma, m)
    y_pred_rand = model_rand.predict(np.concatenate([x_rand, x_new_rand]))
    
    # Визуализация
    plot_results(x_rand, y_rand, x_new_rand, y_new_rand, y_pred_rand,
                f"Линейная регрессия для случайных x на отрезке [{t1}, {t2}]")

if __name__ == "__main__":
    linear_regression_task()