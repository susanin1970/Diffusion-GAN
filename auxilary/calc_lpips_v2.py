"""
Скрипт для подсчета метрики LPIPS (Learned Perceptual Image Patch Similarity)
на основе официальной реализации Zhang et al. (2018), пакет `lpips`.

Отличие от предыдущей версии: используются предобученные калибровочные
линейные слои (обучены авторами на датасете человеческих перцептивных
оценок BAPPS), поэтому итоговое значение лежит в ожидаемом диапазоне
[0, ~1] и сопоставимо со стандартными пороговыми интерпретациями.

Установка зависимости:
    pip install lpips
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import lpips
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def load_image(img_path: Path, size: int = 256) -> torch.Tensor:
    """Загружает изображение и приводит его к тензору в диапазоне [-1, 1]."""
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def load_images_from_folder(folder_path: str, size: int = 256):
    """Загружает все изображения из папки."""
    images, paths = [], []
    folder = Path(folder_path)
    for img_path in sorted(folder.glob("*")):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            try:
                images.append(load_image(img_path, size).squeeze(0))
                paths.append(img_path)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")
    return images, paths


def calculate_lpips_between_pairs(
    folder1: str,
    folder2: str,
    net: str = "alex",
    device: str = "cuda",
    match_by_name: bool = False,
):
    """
    Вычисляет LPIPS между парами изображений из двух папок.

    Args:
        folder1: папка с первым набором изображений (например, реальные данные)
        folder2: папка со вторым набором изображений (например, сгенерированные)
        net: 'alex', 'vgg' или 'squeeze' — сеть для извлечения признаков
        device: 'cuda' или 'cpu'
        match_by_name: True — сравнивать по совпадающим именам файлов,
                       False — сравнивать по порядку (min(len1, len2) пар)

    Returns:
        mean_lpips, std_lpips, distances (np.ndarray)
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        device = "cpu"

    print(f"Загрузка модели LPIPS ({net})...")
    lpips_model = lpips.LPIPS(net=net).to(device)
    lpips_model.eval()

    print(f"\nЗагрузка изображений из {folder1}...")
    images1, paths1 = load_images_from_folder(folder1)
    print(f"Загрузка изображений из {folder2}...")
    images2, paths2 = load_images_from_folder(folder2)

    distances = []

    with torch.no_grad():
        if match_by_name:
            dict2 = {p.name: img for img, p in zip(images2, paths2)}
            pairs_found = 0
            print("\nВычисление LPIPS для пар с одинаковыми именами...")
            for img1, path1 in tqdm(list(zip(images1, paths1))):
                if path1.name in dict2:
                    pairs_found += 1
                    d = lpips_model(
                        img1.unsqueeze(0).to(device),
                        dict2[path1.name].unsqueeze(0).to(device),
                    )
                    distances.append(d.item())
                else:
                    print(f"Предупреждение: не найдена пара для {path1.name}")
            if pairs_found == 0:
                raise ValueError("Не найдено ни одной соответствующей пары изображений")
        else:
            min_len = min(len(images1), len(images2))
            if len(images1) != len(images2):
                print(
                    f"\nВнимание: количество изображений различается "
                    f"({len(images1)} vs {len(images2)}). "
                    f"Будет сравнено первые {min_len} пар"
                )
            print(f"\nВычисление LPIPS для {min_len} пар изображений по порядку...")
            for i in tqdm(range(min_len)):
                d = lpips_model(
                    images1[i].unsqueeze(0).to(device),
                    images2[i].unsqueeze(0).to(device),
                )
                distances.append(d.item())

    distances = np.array(distances)
    return np.mean(distances), np.std(distances), distances


def arguments_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Скрипт для подсчета метрики LPIPS")
    parser.add_argument(
        "-i", "--images_path", type=str, required=True,
        help="Путь к набору реальных изображений",
    )
    parser.add_argument(
        "-gi", "--generated_images_path", type=str, required=True,
        help="Путь к набору сгенерированных изображений",
    )
    parser.add_argument(
        "--net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
        help="Сеть для извлечения признаков (по умолчанию alex, как в оригинальной работе)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--match_by_name", action="store_true",
        help="Сравнивать изображения по совпадающим именам файлов "
             "(по умолчанию — по порядку)",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments_parser()

    mean_lpips, std_lpips, distances = calculate_lpips_between_pairs(
        folder1=args.images_path,
        folder2=args.generated_images_path,
        net=args.net,
        device=args.device,
        match_by_name=args.match_by_name,
    )

    print(f"\n{'=' * 70}")
    print(f"Средний LPIPS: {mean_lpips:.4f} ± {std_lpips:.4f}")
    print(f"Диапазон: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
