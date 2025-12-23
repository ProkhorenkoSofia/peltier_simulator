"""
Реализация метода прогонки (алгоритм Томаса) для решения трехдиагональных систем
"""
import numpy as np


def thomas_algorithm(a, b, c, d):
    """
    Решение трехдиагональной системы уравнений методом прогонки.

    Система: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    для i = 0..n-1, где a[0] = 0, c[n-1] = 0

    Параметры:
    ----------
    a, b, c, d : array-like
        Диагонали матрицы и правая часть

    Возвращает:
    -----------
    x : ndarray
        Решение системы
    """
    n = len(d)

    # Преобразуем в массивы numpy для надежности
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    # Прямой ход прогонки
    cp = np.zeros(n)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # Обратный ход прогонки
    x = np.zeros(n)
    x[-1] = dp[-1]

    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def validate_tridiagonal_system(a, b, c, d, x):
    """
    Проверка решения трехдиагональной системы.

    Параметры:
    ----------
    a, b, c, d : array-like
        Диагонали матрицы и правая часть
    x : array-like
        Предполагаемое решение

    Возвращает:
    -----------
    residual : float
        Максимальная невязка
    """
    n = len(d)
    residual = np.zeros(n)

    for i in range(n):
        lhs = b[i] * x[i]
        if i > 0:
            lhs += a[i] * x[i - 1]
        if i < n - 1:
            lhs += c[i] * x[i + 1]
        residual[i] = abs(lhs - d[i])

    return np.max(residual)
