# python
import re
from typing import List, Tuple, Union

# 3rdparty
import numpy as np


def parse_range(s: Union[str, List[int]]) -> List[int]:
    """Парсит строку с числами или диапазонами, разделенными запятыми, и возвращает список целых чисел.

    Пример: '1,2,5-10' возвращает [1, 2, 5, 6, 7]
    """
    if isinstance(s, list): 
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Парсит строку с 2-вектором из чисел с плавающей точкой в формате 'a,b'.

    Пример:
        '0,1' возвращает (0,1)
    """
    if isinstance(s, tuple): 
        return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(translate: Tuple[float, float], angle: float) -> np.ndarray:
    """Создать матрицу трансформации для поворота и переноса."""
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m
