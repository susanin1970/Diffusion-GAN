import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy


class InceptionV3Classifier(nn.Module):
    """Использует предобученную InceptionV3 для классификации"""

    def __init__(self):
        super().__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.inception.eval()

    def forward(self, x):
        # Выход: [batch_size, 1000] - вероятности классов
        if self.training:
            x, _ = self.inception(x)
        else:
            x = self.inception(x)
        return F.softmax(x, dim=1)


def load_images_from_folder(folder_path, img_size=299):
    """Загружает изображения из папки"""
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images = []
    folder = Path(folder_path)

    print(f"Загрузка изображений из {folder_path}...")
    for img_path in folder.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")

    if len(images) == 0:
        raise ValueError(f"Не найдено изображений в {folder_path}")

    return torch.stack(images)


def get_predictions(model, images, batch_size=32, device="cuda"):
    """
    Получает предсказания модели для изображений

    Returns:
        predictions: numpy array [n_images, 1000] с вероятностями классов
    """
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Получение предсказаний"):
            batch = images[i : i + batch_size].to(device)
            pred = model(batch)
            predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def calculate_inception_score(preds, splits=10):
    """
    Вычисляет Inception Score

    IS = exp(E[KL(p(y|x) || p(y))])

    где:
    - p(y|x) - условное распределение классов для конкретного изображения
    - p(y) - маргинальное распределение классов по всем изображениям
    - KL - дивергенция Кульбака-Лейблера

    Args:
        preds: numpy array [n_images, n_classes] с вероятностями классов
        splits: количество разбиений для усреднения (для стабильности)

    Returns:
        mean_score: среднее значение IS
        std_score: стандартное отклонение IS
    """
    n_images = preds.shape[0]

    # Разбиваем на части для более стабильной оценки
    split_scores = []

    for k in range(splits):
        # Выбираем подмножество изображений
        part = preds[k * (n_images // splits) : (k + 1) * (n_images // splits), :]

        # p(y|x) - условные вероятности (уже есть в preds)
        py_given_x = part

        # p(y) - маргинальное распределение (среднее по всем изображениям)
        py = np.mean(py_given_x, axis=0)

        # KL дивергенция для каждого изображения: KL(p(y|x) || p(y))
        kl_divergences = []
        for i in range(part.shape[0]):
            kl_div = entropy(py_given_x[i], py)
            kl_divergences.append(kl_div)

        # Среднее KL дивергенции
        mean_kl = np.mean(kl_divergences)

        # IS = exp(E[KL])
        split_scores.append(np.exp(mean_kl))

    return np.mean(split_scores), np.std(split_scores)


def calculate_inception_score_alternative(preds, splits=10):
    """
    Альтернативная векторизованная реализация для скорости
    """
    n_images = preds.shape[0]
    split_scores = []

    for k in range(splits):
        part = preds[k * (n_images // splits) : (k + 1) * (n_images // splits), :]

        # p(y|x)
        py_given_x = part

        # p(y)
        py = np.expand_dims(np.mean(py_given_x, axis=0), 0)

        # KL дивергенция: sum(p(y|x) * log(p(y|x) / p(y)))
        kl_div = py_given_x * (np.log(py_given_x + 1e-16) - np.log(py + 1e-16))
        kl_div = np.sum(kl_div, axis=1)

        # IS = exp(mean(KL))
        split_scores.append(np.exp(np.mean(kl_div)))

    return np.mean(split_scores), np.std(split_scores)


def inception_score(images_path, batch_size=32, splits=10, device="cuda"):
    """
    Основная функция для расчета Inception Score

    Args:
        images_path: путь к папке с изображениями
        batch_size: размер батча для обработки
        splits: количество разбиений для усреднения
        device: устройство ('cuda' или 'cpu')

    Returns:
        mean_score: среднее значение IS
        std_score: стандартное отклонение IS
    """
    # Проверка доступности GPU
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        device = "cpu"

    print("Загрузка модели InceptionV3...")
    model = InceptionV3Classifier()

    print(f"\nЗагрузка изображений из {images_path}...")
    images = load_images_from_folder(images_path)
    print(f"Загружено {len(images)} изображений")

    if len(images) < splits:
        print(f"Внимание: изображений ({len(images)}) меньше чем splits ({splits})")
        print(f"Уменьшение splits до {len(images)}")
        splits = len(images)

    print("\nПолучение предсказаний классификатора...")
    predictions = get_predictions(model, images, batch_size, device)

    print("\nВычисление Inception Score...")
    mean_is, std_is = calculate_inception_score_alternative(predictions, splits)

    return mean_is, std_is


def compare_datasets(
    dataset1_path, dataset2_path, batch_size=32, splits=10, device="cuda"
):
    """
    Сравнивает Inception Score для двух датасетов
    """
    print("=" * 60)
    print("СРАВНЕНИЕ INCEPTION SCORE")
    print("=" * 60)

    print("\n[Датасет 1]")
    is1_mean, is1_std = inception_score(dataset1_path, batch_size, splits, device)

    print("\n" + "=" * 60)
    print("\n[Датасет 2]")
    is2_mean, is2_std = inception_score(dataset2_path, batch_size, splits, device)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 60)
    print(f"Датасет 1 IS: {is1_mean:.2f} ± {is1_std:.2f}")
    print(f"Датасет 2 IS: {is2_mean:.2f} ± {is2_std:.2f}")
    print(f"Разница:      {abs(is1_mean - is2_mean):.2f}")

    if is1_mean > is2_mean:
        print(f"\nДатасет 1 лучше на {((is1_mean/is2_mean - 1) * 100):.1f}%")
    else:
        print(f"\nДатасет 2 лучше на {((is2_mean/is1_mean - 1) * 100):.1f}%")

    return (is1_mean, is1_std), (is2_mean, is2_std)

def arguments_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Скрипт для подсчета метрики LPIPS")
    parser.add_argument("-i", "--images_path", type=str, help="Путь к набору реальных изображений")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguments_parser()
    images_path = args.images_path

    try:
        mean_score, std_score = inception_score(
            images_path=images_path,
            batch_size=32,
            splits=10,
            device="cuda",  # или 'cpu'
        )

        print(f"\n{'='*60}")
        print(f"Inception Score: {mean_score:.2f} ± {std_score:.2f}")
        print(f"{'='*60}")
        print("\nИнтерпретация:")
        print("  IS > 10:   Отличное качество и разнообразие")
        print("  IS 5-10:   Хорошее качество")
        print("  IS < 5:    Требуется улучшение")
        print("\nСправка:")
        print("  - Высокий IS означает:")
        print("    1) Изображения четкие (модель уверена в классе)")
        print("    2) Изображения разнообразные (много разных классов)")
        print("  - Максимальный теоретический IS ≈ 1000 (число классов)")

    except Exception as e:
        print(f"Ошибка: {e}")
