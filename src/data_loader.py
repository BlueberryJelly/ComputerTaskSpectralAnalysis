"""Загрузка временного ряда."""
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED_SUFFIXES: frozenset[str] = frozenset({".csv"})


def load_traffic(path: Path) -> np.ndarray:
    """Прочитать CSV с колонкой traffic и вернуть ряд.

    :param path: путь к csv файлу
    :raises FileNotFoundError: файл не существует
    :raises ValueError: недопустимое расширение, отсутствует колонка traffic 
        или в ней есть пропуски
    :return: одномерный массив замеров пак/с
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не существует: {path}")
    if path.suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Файл имеет недопустимое расширение: {path.suffix}")
    frame = pd.read_csv(path)
    if "traffic" not in frame.columns:
        raise ValueError("В файле отсутствует колонка 'traffic'")
    if frame["traffic"].hasnans:
        raise ValueError("Колонка 'traffic' содержит пропуски")

    return frame["traffic"].to_numpy(dtype=float)
