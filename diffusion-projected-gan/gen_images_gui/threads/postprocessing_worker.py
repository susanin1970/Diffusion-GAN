"""Рабочий поток для постобработки изображений."""

import os
import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


def post_processing_algorithm(
    input_image,
    K: Optional[np.ndarray] = None,
    strength: float = 0.8,
    kernel_size_gauss: Tuple[int, int] = (3, 3),
    kernel_size_median: int = 3,
    preserve_color: bool = True,
) -> cv2.Mat:
    """
    Алгоритм пост-обработки изображения согласно блок-схеме

    Параметры:
    input_image - входное изображение (numpy array) или путь к файлу
    K - матрица для фильтра повышения резкости
    strength - величина "силы" нерезкого маскирования
    kernel_size_gauss - размерность ядра фильтра Гаусса
    kernel_size_median - размерность ядра фильтра медианного размытия
    preserve_color - если True, сохраняет цвета изображения

    Возвращает:
    x_g_out - постобработанное изображение
    """
    if isinstance(input_image, str):
        x_g = cv2.imread(input_image)
        if x_g is None:
            raise ValueError(f"Не удалось загрузить изображение: {input_image}")
    else:
        x_g = input_image.copy()

    if not preserve_color and len(x_g.shape) == 3:
        x_g = cv2.cvtColor(x_g, cv2.COLOR_BGR2GRAY)

    if K is None:
        K = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    # Шаг 1: Применение первого шага нерезкого маскирования
    x_g_umb = cv2.GaussianBlur(x_g, (0, 0), sigmaX=1.0)

    # Шаг 2: Применение второго шага нерезкого маскирования
    x_g_ume = cv2.addWeighted(x_g, 1.0 + strength, x_g_umb, -strength, 0)

    # Шаг 3: Применение фильтра увеличения резкости с ядром K
    x_g_sh = cv2.filter2D(x_g_ume, -1, K)

    # Шаг 4: Применение гауссова размытия с ядром kernel_size_gauss
    x_g_gaus = cv2.GaussianBlur(x_g_sh, kernel_size_gauss, sigmaX=0.5)

    # Шаг 5: Применение медианного размытия с ядром kernel_size_median
    x_g_out = cv2.medianBlur(x_g_gaus, kernel_size_median)
    return x_g_out


class PostprocessingWorker(QThread):
    """Рабочий поток для постобработки изображений"""
    
    log_message = Signal(str)
    postprocessing_complete = Signal(str, int)  # Передает путь к директории и количество обработанных изображений
    error_occurred = Signal(str)
    
    def __init__(
        self,
        images_dir: str,
        strength: float,
        kernel_size_gauss: Tuple[int, int],
        kernel_size_median: int,
        preserve_color: bool,
        K: Optional[np.ndarray] = None
    ) -> None:
        super().__init__()
        self.images_dir: str = images_dir
        self.strength: float = strength
        self.kernel_size_gauss: Tuple[int, int] = kernel_size_gauss
        self.kernel_size_median: int = kernel_size_median
        self.preserve_color: bool = preserve_color
        self.K: Optional[np.ndarray] = K
        
    def run(self) -> None:
        try:
            # Найти все изображения в директории
            image_extensions = ['*.png', '*.jpg', '*.jpeg']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
            
            image_files = sorted(image_files)
            
            if not image_files:
                self.error_occurred.emit('Изображения не найдены в указанной директории')
                return
            
            total_count = len(image_files)
            self.log_message.emit(f'Найдено изображений для обработки: {total_count}')
            
            processed_count = 0
            for idx, image_path in enumerate(image_files):
                try:
                    self.log_message.emit(f'Обработка изображения {idx+1}/{total_count}: {os.path.basename(image_path)}')
                    
                    # Загрузить изображение
                    original = cv2.imread(image_path)
                    if original is None:
                        self.log_message.emit(f'Предупреждение: не удалось загрузить {image_path}')
                        continue
                    
                    # Применить постобработку
                    result = post_processing_algorithm(
                        original,
                        K=self.K,
                        strength=self.strength,
                        kernel_size_gauss=self.kernel_size_gauss,
                        kernel_size_median=self.kernel_size_median,
                        preserve_color=self.preserve_color
                    )
                    
                    # Сохранить обработанное изображение (перезаписать исходное)
                    cv2.imwrite(image_path, result)
                    processed_count += 1
                    self.log_message.emit(f'Обработано изображение: {os.path.basename(image_path)}')
                    
                except Exception as e:
                    self.log_message.emit(f'Ошибка при обработке {os.path.basename(image_path)}: {str(e)}')
            
            self.log_message.emit(f'Постобработка завершена! Обработано изображений: {processed_count} из {total_count}')
            self.postprocessing_complete.emit(self.images_dir, processed_count)
            
        except Exception as e:
            self.error_occurred.emit(f'Ошибка при постобработке: {str(e)}')
