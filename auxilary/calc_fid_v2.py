"""
Скрипт для подсчета метрики FID (Fréchet Inception Distance)
на основе канонической сети FID Inception (pytorch-fid, Seitzer,
порт официальной TensorFlow-реализации Heusel et al., 2017).

Отличие от v2: реализована собственная загрузка изображений с явным
приведением к единому размеру (Resize) перед батчингом, поскольку
низкоуровневый API pytorch_fid (calculate_fid_given_paths) не выполняет
ресайз самостоятельно и падает с RuntimeError при разноразмерных
изображениях в датасете (что типично для наборов реальных фотографий,
в отличие от сгенерированных GAN-изображений фиксированного размера).

Установка зависимости:
    pip install pytorch-fid
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy import linalg

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance


class ImageDataset(Dataset):
    """Датасет с приведением всех изображений к единому размеру."""

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, folder_path: str, resize_to: int = 299):
        self.files = sorted(
            [
                p
                for p in Path(folder_path).glob("*")
                if p.suffix.lower() in self.IMG_EXTENSIONS
            ]
        )
        if len(self.files) == 0:
            raise ValueError(f"Не найдено изображений в {folder_path}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((resize_to, resize_to)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def get_activations(
    folder_path: str,
    model: InceptionV3,
    batch_size: int,
    device: str,
    resize_to: int = 299,
    num_workers: int = 4,
) -> np.ndarray:
    """Извлекает признаки InceptionV3 для всех изображений в папке."""
    dataset = ImageDataset(folder_path, resize_to=resize_to)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    activations = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Извлечение признаков ({Path(folder_path).name})"):
            batch = batch.to(device)
            pred = model(batch)[0]
            # pool3-признаки имеют форму (batch, dims, 1, 1) -> сжимаем
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            activations.append(pred.squeeze(3).squeeze(2).cpu().numpy())

    return np.concatenate(activations, axis=0)


def calculate_statistics(features: np.ndarray):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(
    real_images_path: str,
    generated_images_path: str,
    batch_size: int = 32,
    device: str = "cuda",
    dims: int = 2048,
    resize_to: int = 299,
    num_workers: int = 4,
) -> float:
    """
    Вычисляет FID между двумя наборами изображений разного исходного
    разрешения, приводя их к единому размеру перед подачей в модель.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        device = "cpu"

    if dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
        raise ValueError(
            f"dims должен быть одним из {list(InceptionV3.BLOCK_INDEX_BY_DIM.keys())}"
        )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    print(f"\nИзвлечение признаков из реальных изображений ({real_images_path})...")
    real_features = get_activations(
        real_images_path, model, batch_size, device, resize_to, num_workers
    )
    print(f"\nИзвлечение признаков из сгенерированных изображений ({generated_images_path})...")
    gen_features = get_activations(
        generated_images_path, model, batch_size, device, resize_to, num_workers
    )

    print(f"\nРеальных изображений: {real_features.shape[0]}, "
          f"сгенерированных: {gen_features.shape[0]}, размерность признаков: {dims}")

    if real_features.shape[0] < dims or gen_features.shape[0] < dims:
        print(
            f"ВНИМАНИЕ: объём одной из выборок меньше размерности признаков ({dims}). "
            f"Ковариационная матрица может быть вырожденной. "
            f"Рекомендуется использовать --dims 768 или меньше."
        )

    mu_real, sigma_real = calculate_statistics(real_features)
    mu_gen, sigma_gen = calculate_statistics(gen_features)

    print("\nВычисление FID...")
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return fid_score


def arguments_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Скрипт для подсчета метрики FID")
    parser.add_argument(
        "-i", "--images_path", type=str, required=True,
        help="Путь к набору реальных изображений",
    )
    parser.add_argument(
        "-gi", "--generated_images_path", type=str, required=True,
        help="Путь к набору сгенерированных изображений",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dims", type=int, default=2048,
        help="Размерность признаков InceptionV3 (2048 — стандарт; "
             "используйте 768/192/64 при малом объёме выборки)",
    )
    parser.add_argument(
        "--resize_to", type=int, default=299,
        help="Размер, к которому приводятся все изображения перед подачей в модель",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = arguments_parser()

    fid = calculate_fid(
        real_images_path=args.images_path,
        generated_images_path=args.generated_images_path,
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims,
        resize_to=args.resize_to,
        num_workers=args.num_workers,
    )

    print(f"\n{'=' * 50}")
    print(f"Fréchet Inception Distance (FID): {fid:.2f}")
    print(f"{'=' * 50}")
    print("\nИнтерпретация:")
    print(" FID < 10: Отличное качество")
    print(" FID 10-50: Хорошее качество")
    print(" FID > 50: Требуется улучшение")


if __name__ == "__main__":
    main()
