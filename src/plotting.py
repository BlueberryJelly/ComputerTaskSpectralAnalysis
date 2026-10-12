"""Модуль для построения и сохранения графиков анализа трафика."""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_plot(path: Path) -> None:
    """Сохранить текущую фигуру в файл и закрыть её для освобождения памяти.

    :param path: файл для сохранения графика
    """
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_series(series: np.ndarray, path: Path) -> None:
    """Построить и сохранить график временного ряда.

    :param series: одномерный массив значений интенсивности трафика, пакетов/с
    :param path: файл для сохранения графика
    """
    plt.figure(figsize=(10, 4))
    plt.plot(series, lw=0.9)
    plt.title("Интенсивность трафика")
    plt.xlabel("временная точка")
    plt.ylabel("пакетов/с")
    plt.grid(alpha=0.3)
    _save_plot(path)


def plot_spectrum(amplitudes: np.ndarray, path: Path) -> None:
    """Построить и сохранить гистограмму амплитудного спектра A_k для k = 1..K.

    Нулевая гармоника (постоянная составляющая при k=0) исключена из графика.

    :param amplitudes: массив амплитуд гармоник, где индекс 0 соответствует k=0
    :param path: файл для сохранения графика
    """
    plt.figure(figsize=(10, 4))
    k_values = np.arange(1, amplitudes.size)
    plt.bar(k_values, amplitudes[1:])
    plt.title("Амплитудный спектр")
    plt.xlabel("номер гармоники k")
    plt.ylabel("A_k")
    _save_plot(path)


def plot_approximation(series: np.ndarray, approx: np.ndarray, order: int,
                       path: Path) -> None:
    """Построить и сохранить график исходного ряда и его тригонометрического приближения.

    :param series: исходный одномерный массив временного ряда
    :param approx: массив значений приближения, тригонометрического многочлена
    :param order: порядок приближения, количество учитываемых гармоник K
    :param path: файл для сохранения графика
    """
    plt.figure(figsize=(10, 4))
    plt.plot(series, color="0.7", label="исходный ряд")
    plt.plot(approx, color="tab:red", label=f"K={order}")
    plt.title(f"Тригонометрический многочлен, K={order}")
    plt.xlabel("временная точка")
    plt.ylabel("пакетов/с")
    plt.legend()
    _save_plot(path)
