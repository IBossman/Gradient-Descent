import numpy as np

def gradient(y_true: int, y_pred: float, x: np.array) -> np.array:
    """
    y_true - истинное значение ответа для объекта x
    y_pred - значение степени принадлежности объекта x классу 1, предсказанное нашей моделью
    x - вектор признакового описания данного объекта

    На выходе ожидается получить вектор частных производных H по параметрам модели, предсказавшей значение y_pred
    Обратите внимание, что размерность этого градиента должна получиться на единицу больше размерности x засчет свободного коэффициента a0
    """
    grad = x * ((1 - y_true) * y_pred - y_true * (1 - y_pred))
    grad = np.append(grad, ((1 - y_true) * y_pred - y_true * (1 - y_pred)))
    return grad


# Функция обновления весов
def update(alpha: np.array, gradient: np.array, lr: float):
    """
    alpha: текущее приближения вектора параметров модели
    gradient: посчитанный градиент по параметрам модели
    lr: learning rate, множитель перед градиентом в формуле обновления параметров
    """
    alpha_new = alpha - gradient * lr
    return alpha_new


# функция тренировки модели
def train(
    alpha0: np.array, x_train: np.array, y_train: np.array, lr: float, num_epoch: int
):
    """
    alpha0 - начальное приближение параметров модели
    x_train - матрица объект-признак обучающей выборки
    y_train - верные ответы для обучающей выборки
    lr - learning rate, множитель перед градиентом в формуле обновления параметров
    num_epoch - количество эпох обучения, то есть полных 'проходов' через весь датасет
    """
    
    alpha = alpha0.copy()
    for epo in range(num_epoch):
        for i, x in enumerate(x_train):

            x_sigma = np.dot(x, alpha[:len(alpha) - 1]) + alpha[len(alpha) - 1]
            sigma = 1 / (1 +  np.exp(- x_sigma))
            alpha = update(alpha, gradient(y_train[i], sigma, x), lr)

    return alpha