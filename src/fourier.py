"""Ручное ДПФ (без FFT), поиск пиков и тригонометрический многочлен."""
import numpy as np


def _angles(orders: np.ndarray, n: int) -> np.ndarray:
    """Вычислить матрицу углов 2*pi*k*j/N для ДПФ.

    Строки матрицы соответствуют номерам гармоник k, столбцы - индексам
    временных точек j.

    :param orders: массив номеров гармоник k
    :param n: длина временного ряда N
    :return: матрица углов размера (len(orders), n)
    """
    return 2 * np.pi * orders[:, None] * np.arange(n) / n


def fourier_coefficients(series: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Вычислить коэффициенты Фурье a_k и b_k для k = 0..k_max по формулам ДПФ.

    :param series: одномерный массив временного ряда
    :param k_max: максимальный номер гармоники для вычисления
    :return: кортеж (a, b), где a и b - массивы коэффициентов длиной k_max + 1
    """
    n = series.size
    angle = _angles(np.arange(k_max + 1), n)
    return 2 / n * (np.cos(angle) @ series), 2 / n * (np.sin(angle) @ series)


def find_peaks(amplitudes: np.ndarray, factor: float = 5.0) -> list[int]:
    """Найти номера значимых гармоник, где амплитуда превышает порог.

    :param amplitudes: массив амплитуд гармоник, индекс 0 соответствует k=0
    :param factor: множитель для порога относительно медианы, по умолчанию 5.0
    :return: список номеров гармоник k >= 1, где A_k > factor * медиана
    """
    threshold = factor * np.median(amplitudes[1:])
    return [k for k in range(1, amplitudes.size) if amplitudes[k] > threshold]


def trig_polynomial(a: np.ndarray, b: np.ndarray, n: int, order: int) -> np.ndarray:
    """Построить тригонометрический многочлен (приближение ряда).

    :param a: массив коэффициентов a_k длины >= order + 1
    :param b: массив коэффициентов b_k длины >= order + 1
    :param n: длина результирующего временного ряда
    :param order: порядок приближения, количество учитываемых гармоник
    :return: одномерный массив значений приближения длиной n
    """
    angle = _angles(np.arange(1, order + 1), n)
    return a[0] / 2 + a[1 : order + 1] @ np.cos(angle) + b[1 : order + 1] @ np.sin(angle)
