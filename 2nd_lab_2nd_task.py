import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None    # Коэффициенты для x1, x2, ... (аналог model.coef_ в sklearn)
        self.intercept_ = None  # Свободный член (аналог model.intercept_)
        
    def fit(self, X, y):
        """Обучение модели методом наименьших квадратов"""
        X = np.array(X)
        y = np.array(y)
        # Добавляем столбец единиц для свободного члена
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Вычисляем коэффициенты: theta = (XᵀX)⁻¹Xᵀy
        theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        
    def predict(self, X):
        """Предсказание значений y по входным данным X"""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Модель еще не обучена! Сначала вызовите fit().")
        return self.intercept_ + np.dot(X, self.coef_)
    
    def score(self, X, y):
        """Вычисление коэффициента детерминации R²"""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1 - (ss_res / ss_total)

def generate_sample(a1, a2, b, sigma, n, t1, t2, s1, s2):
    """Генерация выборки с двумя признаками x1 и x2"""
    x1 = np.random.uniform(t1, t2, n)
    x2 = np.random.uniform(s1, s2, n)
    X = np.column_stack((x1, x2))
    y_true = a1 * x1 + a2 * x2 + b
    y_noisy = y_true + np.random.normal(0, sigma, n)
    return X, y_noisy

def plot_results_3d(X, y, X_new, y_new, y_pred, model, title):
    """Визуализация результатов в 3D"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Исходные данные
    ax.scatter(X[:, 0], X[:, 1], y, c='blue', label='Исходная выборка', alpha=0.6)
    
    # Дополнительные данные
    ax.scatter(X_new[:, 0], X_new[:, 1], y_new, c='red', label='Дополнительная выборка', alpha=0.6)
    
    # Предсказания модели (плоскость)
    x1_grid = np.linspace(min(X[:, 0].min(), X_new[:, 0].min()), 
                         max(X[:, 0].max(), X_new[:, 0].max()), 20)
    x2_grid = np.linspace(min(X[:, 1].min(), X_new[:, 1].min()), 
                         max(X[:, 1].max(), X_new[:, 1].max()), 20)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
    y_mesh = model.intercept_ + model.coef_[0] * x1_mesh + model.coef_[1] * x2_mesh
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh, color='green', alpha=0.3, label='Плоскость регрессии')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.legend()
    plt.show()

def multivariate_regression_task():
    print("=== Многомерная линейная регрессия ===")
    
    # 1) Ввод параметров
    a1 = float(input("Введите коэффициент a1: "))
    a2 = float(input("Введите коэффициент a2: "))
    b = float(input("Введите свободный член b: "))
    sigma = float(input("Введите стандартное отклонение sigma: "))
    n = int(input("Введите размер выборки n: "))
    m = int(input("Введите размер дополнительной выборки m: "))
    t1, t2 = map(float, input("Введите границы для x1 [t1, t2]: ").split())
    s1, s2 = map(float, input("Введите границы для x2 [s1, s2]: ").split())
    
    # 2) Генерация и обучение
    X, y = generate_sample(a1, a2, b, sigma, n, t1, t2, s1, s2)
    model = MyLinearRegression()
    model.fit(X, y)
    print(f"Оценки коэффициентов: a1* = {model.coef_[0]:.3f}, a2* = {model.coef_[1]:.3f}, b* = {model.intercept_:.3f}")
    print(f"R²: {model.score(X, y):.3f}")
    
    # 3) Дополнительная выборка
    X_new, y_new = generate_sample(a1, a2, b, sigma, m, t1, t2, s1, s2)
    y_pred = model.predict(X_new)
    print("\nСравнение предсказаний и реальных значений (первые 5 точек):")
    for i in range(5):
        print(f"x1={X_new[i, 0]:.2f}, x2={X_new[i, 1]:.2f} | y_true={y_new[i]:.2f}, y_pred={y_pred[i]:.2f}")
    
    # 4) Визуализация
    plot_results_3d(X, y, X_new, y_new, y_pred, model,
                   f"Многомерная регрессия: y = {a1:.1f}x1 + {a2:.1f}x2 + {b:.1f}")

if __name__ == "__main__":
    multivariate_regression_task()