"""Спектральный анализ сетевого трафика."""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.data_loader import load_traffic
from src.fourier import find_peaks, fourier_coefficients, trig_polynomial
from src.plotting import plot_approximation, plot_series, plot_spectrum

K_MAX = 50
ORDERS = (2, 20)


def run(csv_path: Path, out_dir: Path) -> None:
    """Выполнить спектральный анализ ряда трафика и сохранить результаты.

    :param csv_path: путь к входному CSV-файлу с временным рядом трафика
    :param out_dir: путь к директории для сохранения графиков
    :raises FileNotFoundError: если csv_path не существует
    :raises ValueError: если данные в CSV некорректны
    """
    series = load_traffic(csv_path)
    a, b = fourier_coefficients(series, K_MAX)
    amplitudes = np.hypot(a, b)

    print(f"a0/2 = {a[0] / 2:.2f}")
    print("k      a_k       b_k       A_k")
    for k in range(1, K_MAX + 1):
        print(f"{k:<4}{a[k]:9.3f}{b[k]:10.3f}{amplitudes[k]:10.3f}")
    print("Аномальные k:", find_peaks(amplitudes))

    plot_series(series, out_dir / "fig1_series.png")
    plot_spectrum(amplitudes, out_dir / "fig2_spectrum.png")
    for order in ORDERS:
        approx = trig_polynomial(a, b, series.size, order)
        plot_approximation(series, approx, order, out_dir / f"fig3_K{order}.png")


def main() -> int:
    """Разобрать аргументы командной строки и запустить анализ.

    :return: 0 при успешном завершении, 1 при ошибке ввода-вывода
             или некорректных данных
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="путь к statistics.csv")
    parser.add_argument("-o", "--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    try:
        run(args.csv, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
