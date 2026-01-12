import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS)
    Использует предобученные сети для измерения перцептивного расстояния
    """

    def __init__(self, net="alex", use_dropout=True):
        """
        Args:
            net: 'alex' (AlexNet), 'vgg' (VGG16), или 'squeeze' (SqueezeNet)
            use_dropout: использовать ли dropout слои
        """
        super().__init__()
        self.net = net

        if net == "alex":
            self.model = self._get_alexnet()
            self.layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
            self.channels = [64, 192, 384, 256, 256]
        elif net == "vgg":
            self.model = self._get_vgg()
            self.layers = ["conv1_2", "conv2_2", "conv3_2", "conv4_2", "conv5_2"]
            self.channels = [64, 128, 256, 512, 512]
        elif net == "squeeze":
            self.model = self._get_squeezenet()
            self.layers = [
                "conv1",
                "conv2",
                "conv3",
                "conv4",
                "conv5",
                "conv6",
                "conv7",
            ]
            self.channels = [64, 128, 256, 384, 384, 512, 512]
        else:
            raise ValueError(f"Неизвестная сеть: {net}")

        # Линейные слои для калибровки
        self.linear_layers = nn.ModuleList(
            [nn.Conv2d(ch, 1, 1, bias=False) for ch in self.channels]
        )

        # Инициализация весов
        for layer in self.linear_layers:
            layer.weight.data.fill_(1.0)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _get_alexnet(self):
        """Загружает AlexNet и модифицирует для извлечения признаков"""
        alexnet = models.alexnet(pretrained=True)
        layers = []

        # Conv1
        layers.append(
            nn.Sequential(
                alexnet.features[0],  # Conv2d
                alexnet.features[1],  # ReLU
            )
        )

        # Conv2
        layers.append(
            nn.Sequential(
                alexnet.features[2],  # MaxPool2d
                alexnet.features[3],  # Conv2d
                alexnet.features[4],  # ReLU
            )
        )

        # Conv3
        layers.append(
            nn.Sequential(
                alexnet.features[5],  # MaxPool2d
                alexnet.features[6],  # Conv2d
                alexnet.features[7],  # ReLU
            )
        )

        # Conv4
        layers.append(
            nn.Sequential(
                alexnet.features[8],  # Conv2d
                alexnet.features[9],  # ReLU
            )
        )

        # Conv5
        layers.append(
            nn.Sequential(
                alexnet.features[10],  # Conv2d
                alexnet.features[11],  # ReLU
            )
        )

        return nn.ModuleList(layers)

    def _get_vgg(self):
        """Загружает VGG16"""
        vgg = models.vgg16(pretrained=True)
        layers = []

        # Conv1_2
        layers.append(nn.Sequential(*list(vgg.features[:4])))
        # Conv2_2
        layers.append(nn.Sequential(*list(vgg.features[4:9])))
        # Conv3_2
        layers.append(nn.Sequential(*list(vgg.features[9:16])))
        # Conv4_2
        layers.append(nn.Sequential(*list(vgg.features[16:23])))
        # Conv5_2
        layers.append(nn.Sequential(*list(vgg.features[23:30])))

        return nn.ModuleList(layers)

    def _get_squeezenet(self):
        """Загружает SqueezeNet"""
        squeeze = models.squeezenet1_1(pretrained=True)
        layers = []

        layers.append(nn.Sequential(*list(squeeze.features[:2])))  # Conv1
        layers.append(nn.Sequential(*list(squeeze.features[2:5])))  # Conv2
        layers.append(nn.Sequential(*list(squeeze.features[5:8])))  # Conv3
        layers.append(nn.Sequential(*list(squeeze.features[8:10])))  # Conv4
        layers.append(nn.Sequential(*list(squeeze.features[10:11])))  # Conv5
        layers.append(nn.Sequential(*list(squeeze.features[11:12])))  # Conv6
        layers.append(nn.Sequential(*list(squeeze.features[12:])))  # Conv7

        return nn.ModuleList(layers)

    def forward(self, img1, img2, normalize=True):
        """
        Вычисляет LPIPS расстояние между двумя изображениями

        Args:
            img1, img2: тензоры [B, 3, H, W] в диапазоне [-1, 1]
            normalize: нормализовать ли признаки

        Returns:
            lpips_distance: тензор [B] с расстояниями
        """
        # Извлекаем признаки из каждого слоя
        feats1 = self._extract_features(img1)
        feats2 = self._extract_features(img2)

        diffs = []
        for i, (f1, f2) in enumerate(zip(feats1, feats2)):
            # Нормализация по каналам
            if normalize:
                f1 = f1 / (torch.sqrt(torch.sum(f1**2, dim=1, keepdim=True)) + 1e-10)
                f2 = f2 / (torch.sqrt(torch.sum(f2**2, dim=1, keepdim=True)) + 1e-10)

            # Квадрат разности
            diff = (f1 - f2) ** 2

            # Применяем линейный слой
            diff = self.linear_layers[i](diff)

            # Усредняем по пространственным измерениям
            diff = torch.mean(diff, dim=[2, 3], keepdim=True)
            diffs.append(diff)

        # Суммируем по всем слоям
        lpips_distance = sum(diffs)

        return lpips_distance.squeeze()

    def _extract_features(self, x):
        """Извлекает признаки из всех слоев"""
        features = []
        for layer in self.model:
            x = layer(x)
            features.append(x)
        return features


def load_image_pair(img1_path, img2_path, size=256):
    """Загружает пару изображений"""
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ]
    )

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)

    return img1_tensor, img2_tensor


def load_images_from_folder(folder_path, size=256):
    """Загружает все изображения из папки"""
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    images = []
    paths = []
    folder = Path(folder_path)

    for img_path in sorted(folder.glob("*")):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                images.append(img_tensor)
                paths.append(img_path)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")

    return images, paths


def calculate_lpips_between_pairs(
    folder1, folder2, net="alex", batch_size=1, device="cuda", match_by_name=True
):
    """
    Вычисляет LPIPS между парами изображений из двух папок

    Args:
        folder1: папка с первым набором изображений
        folder2: папка со вторым набором изображений
        net: 'alex', 'vgg' или 'squeeze'
        batch_size: размер батча
        device: 'cuda' или 'cpu'
        match_by_name: True - сравнивать по именам файлов, False - по порядку

    Returns:
        mean_lpips: среднее LPIPS расстояние
        std_lpips: стандартное отклонение
        distances: список всех расстояний
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        device = "cpu"

    print(f"Загрузка модели LPIPS ({net})...")
    lpips_model = LPIPS(net=net).to(device)
    lpips_model.eval()

    print(f"\nЗагрузка изображений из {folder1}...")
    images1, paths1 = load_images_from_folder(folder1)

    print(f"Загрузка изображений из {folder2}...")
    images2, paths2 = load_images_from_folder(folder2)

    distances = []

    if match_by_name:
        # Сравниваем по именам файлов (старое поведение)
        dict2 = {p.name: (img, p) for img, p in zip(images2, paths2)}
        pairs_found = 0

        print(f"\nВычисление LPIPS для пар с одинаковыми именами...")

        with torch.no_grad():
            for img1, path1 in tqdm(zip(images1, paths1), total=len(images1)):
                name1 = path1.name

                if name1 in dict2:
                    img2, path2 = dict2[name1]
                    pairs_found += 1

                    img1_batch = img1.unsqueeze(0).to(device)
                    img2_batch = img2.unsqueeze(0).to(device)

                    distance = lpips_model(img1_batch, img2_batch)
                    distances.append(distance.item())
                else:
                    print(f"Предупреждение: не найдена пара для {name1}")

        if pairs_found == 0:
            raise ValueError("Не найдено ни одной соответствующей пары изображений")
    else:
        # Сравниваем по порядку (новое поведение)
        min_len = min(len(images1), len(images2))

        if len(images1) != len(images2):
            print(
                f"\nВнимание: количество изображений различается ({len(images1)} vs {len(images2)})"
            )
            print(f"Будет сравнено первые {min_len} пар")

        print(f"\nВычисление LPIPS для {min_len} пар изображений по порядку...")

        with torch.no_grad():
            for i in tqdm(range(min_len)):
                img1 = images1[i].unsqueeze(0).to(device)
                img2 = images2[i].unsqueeze(0).to(device)

                distance = lpips_model(img1, img2)
                distances.append(distance.item())

    distances = np.array(distances)
    return np.mean(distances), np.std(distances), distances


def calculate_lpips_diversity(folder, net="alex", num_samples=1000, device="cuda"):
    """
    Вычисляет среднее LPIPS расстояние между случайными парами изображений
    (оценка разнообразия датасета)

    Args:
        folder: папка с изображениями
        net: 'alex', 'vgg' или 'squeeze'
        num_samples: количество случайных пар для оценки
        device: 'cuda' или 'cpu'

    Returns:
        mean_lpips: среднее LPIPS расстояние
        std_lpips: стандартное отклонение
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        device = "cpu"

    print(f"Загрузка модели LPIPS ({net})...")
    lpips_model = LPIPS(net=net).to(device)
    lpips_model.eval()

    print(f"\nЗагрузка изображений из {folder}...")
    images, paths = load_images_from_folder(folder)

    if len(images) < 2:
        raise ValueError("Недостаточно изображений для сравнения")

    # Генерируем случайные пары
    n_images = len(images)
    max_pairs = n_images * (n_images - 1) // 2
    num_samples = min(num_samples, max_pairs)

    print(f"\nВычисление LPIPS для {num_samples} случайных пар...")

    distances = []
    rng = np.random.RandomState(42)

    with torch.no_grad():
        for _ in tqdm(range(num_samples)):
            # Выбираем две случайные картинки
            idx1, idx2 = rng.choice(n_images, size=2, replace=False)

            img1 = images[idx1].unsqueeze(0).to(device)
            img2 = images[idx2].unsqueeze(0).to(device)

            distance = lpips_model(img1, img2)
            distances.append(distance.item())

    distances = np.array(distances)
    return np.mean(distances), np.std(distances)


def compare_single_pair(img1_path, img2_path, net="alex", device="cuda"):
    """
    Сравнивает две конкретные картинки
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        device = "cpu"

    print(f"Загрузка модели LPIPS ({net})...")
    lpips_model = LPIPS(net=net).to(device)
    lpips_model.eval()

    print(f"\nЗагрузка изображений...")
    img1, img2 = load_image_pair(img1_path, img2_path)

    print("Вычисление LPIPS...")
    with torch.no_grad():
        img1 = img1.to(device)
        img2 = img2.to(device)
        distance = lpips_model(img1, img2)

    return distance.item()

def arguments_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Скрипт для подсчета метрики LPIPS")
    parser.add_argument("-i", "--images_path", type=str, help="Путь к набору реальных изображений")
    parser.add_argument("-gi", "--generated_images_path", type=str, help="Путь к набору сгенерированных изображений")
    args = parser.parse_args()
    return args

def main() -> None:
    args = arguments_parser()
    folder1 = args.images_path
    folder2 = args.enerated_images_path
    
    try:
        mean_lpips, std_lpips, distances = calculate_lpips_between_pairs(
            folder1=folder1,
            folder2=folder2,
            net="alex",
            device="cuda",
            match_by_name=False,  # Сравнивать по именам файлов
        )

        print(f"\n{'='*70}")
        print(f"Средний LPIPS: {mean_lpips:.4f} ± {std_lpips:.4f}")
        print(f"{'='*70}")

    except Exception as e:
        print(f"Ошибка: {e}")

    print("\n" + "=" * 70)
    print("ПРИМЕР 2: Оценка разнообразия датасета")
    print("=" * 70)

    try:
        mean_lpips, std_lpips = calculate_lpips_diversity(
            folder=folder1,
            net="alex",
            num_samples=500,
            device="cuda",
        )

        print(f"\n{'='*70}")
        print(f"Средний LPIPS (разнообразие): {mean_lpips:.4f} ± {std_lpips:.4f}")
        print(f"{'='*70}")
        print("\nИнтерпретация разнообразия:")
        print("  Высокий LPIPS = высокое разнообразие")
        print("  Низкий LPIPS = изображения похожи друг на друга")

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
