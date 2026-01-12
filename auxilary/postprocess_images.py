# python
import os
import argparse

# 3rdparty
import cv2
import numpy as np


def post_processing_algorithm(
    input_image,
    K=None,
    strength=0.8,
    kernel_size_gauss=(3, 3),
    kernel_size_median=3,
    preserve_color=True,
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


def process_image_pipeline(
    input_path, output_path=None, **kwargs
) -> tuple[cv2.Mat, cv2.Mat]:
    """
    Полный конвейер обработки изображения с сохранением результата

    Параметры:
    input_path - путь к входному изображению
    output_path - путь для сохранения результата (опционально)
    **kwargs - дополнительные параметры для post_processing_algorithm
    """
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError(f"Не удалось загрузить изображение: {input_path}")

    print("Начало пост-обработки изображения...")

    result = post_processing_algorithm(original, **kwargs)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Постобработанное изображение сохранено: {output_path}")

    return original, result


def arguments_parser() -> argparse.Namespace:
    """Парсер аргументов"""
    parser = argparse.ArgumentParser(
        description="Скрипт для выполнения постобработки сгенерированных с помощью натренированной модифицированной Diffusion-GAN изображений"
    )
    parser.add_argument(
        "-i",
        "--images_path",
        type=str,
        help="Путь к папке с исходными сгенерированными изображениями",
    )
    parser.add_argument(
        "-pi",
        "--postprocessed_images_path",
        type=str,
        help="Путь к папке с постобработанными сгенерированными изображениями",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = arguments_parser()
    images_path = args.images_path
    postprocessed_images_path = args.postprocessed_images_path
    
    images_path = r"E:\Datasets\DIFFGAN_GENERATED_IMAGES\images_for_training"
    postprocessed_images_path = (
        "E:\Datasets\DIFFGAN_GENERATED_IMAGES\images_for_training_postprocessed"
    )

    if not os.path.exists(postprocessed_images_path):
        os.makedirs(postprocessed_images_path)

    for image in os.listdir(images_path):
        try:
            path_to_input_image = os.path.join(images_path, image)
            path_to_output_image = os.path.join(postprocessed_images_path, image)
            original, processed = process_image_pipeline(
                path_to_input_image,
                path_to_output_image,
                strength=0.8,
                kernel_size_gauss=(3, 3),
                kernel_size_median=3,
            )

        except Exception as e:
            print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()
