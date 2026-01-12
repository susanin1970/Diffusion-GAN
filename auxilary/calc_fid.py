import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy import linalg
from pathlib import Path
from tqdm import tqdm


class InceptionV3(nn.Module):
    """Использует предобученную InceptionV3 для извлечения признаков"""

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()  # Удаляем последний FC слой
        inception.aux_logits = False  # Отключаем вспомогательные выходы
        self.inception = inception
        self.inception.eval()

    def forward(self, x):
        # Изменяем размер входа под требования InceptionV3 (299x299)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = nn.functional.interpolate(
                x, size=(299, 299), mode="bilinear", align_corners=False
            )

        # Выход: [batch_size, 2048]
        x = self.inception(x)
        return x


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

    for img_path in folder.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")

    return torch.stack(images)


def extract_features(model, images, batch_size=32, device="cuda"):
    """Извлекает признаки из изображений"""
    model = model.to(device)
    model.eval()

    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Извлечение признаков"):
            batch = images[i : i + batch_size].to(device)
            feat = model(batch)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def calculate_statistics(features):
    """Вычисляет среднее и ковариационную матрицу"""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Вычисляет расстояние Фреше между двумя многомерными гауссианами

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Произведение ковариационных матриц
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Численная стабильность
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Удаляем мнимую часть (если появилась из-за погрешностей)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Мнимая составляющая слишком велика: {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return fid


def calculate_fid(
    real_images_path, generated_images_path, batch_size=32, device="cuda"
):
    """
    Основная функция для расчета FID

    Args:
        real_images_path: путь к папке с реальными изображениями
        generated_images_path: путь к папке с сгенерированными изображениями
        batch_size: размер батча для обработки
        device: устройство ('cuda' или 'cpu')

    Returns:
        fid_score: значение метрики FID
    """
    # Проверка доступности GPU
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        device = "cpu"

    print("Загрузка модели InceptionV3...")
    model = InceptionV3()

    print(f"\nЗагрузка реальных изображений из {real_images_path}...")
    real_images = load_images_from_folder(real_images_path)
    print(f"Загружено {len(real_images)} реальных изображений")

    print(f"\nЗагрузка сгенерированных изображений из {generated_images_path}...")
    gen_images = load_images_from_folder(generated_images_path)
    print(f"Загружено {len(gen_images)} сгенерированных изображений")

    print("\nИзвлечение признаков из реальных изображений...")
    real_features = extract_features(model, real_images, batch_size, device)

    print("\nИзвлечение признаков из сгенерированных изображений...")
    gen_features = extract_features(model, gen_images, batch_size, device)

    print("\nВычисление статистик...")
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_gen, sigma_gen = calculate_statistics(gen_features)

    print("\nВычисление FID...")
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return fid_score

def arguments_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Скрипт для подсчета метрики LPIPS")
    parser.add_argument("-i", "--images_path", type=str, help="Путь к набору реальных изображений")
    parser.add_argument("-gi", "--generated_images_path", type=str, help="Путь к набору сгенерированных изображений")
    args = parser.parse_args()
    return args

def main() -> None:
    args = arguments_parser()
    real_path = args.images_path
    generated_path = args.generated_images_path

    try:
        fid = calculate_fid(
            real_images_path=real_path,
            generated_images_path=generated_path,
            batch_size=32,
            device="cuda",  # или 'cpu'
        )

        print(f"\n{'='*50}")
        print(f"Fréchet Inception Distance (FID): {fid:.2f}")
        print(f"{'='*50}")
        print("\nИнтерпретация:")
        print("  FID < 10:  Отличное качество")
        print("  FID 10-50: Хорошее качество")
        print("  FID > 50:  Требуется улучшение")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()