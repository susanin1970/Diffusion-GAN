# python
import os
import time
from typing import List, Optional, Tuple

# 3rdparty
import dnnlib
import legacy
import numpy as np
import PIL.Image
import torch
from PySide6.QtCore import QThread, Signal

# project
from ..utils import make_transform


class GenerationWorker(QThread):
    """Рабочий поток для генерации изображений"""
    
    log_message = Signal(str)
    generation_complete = Signal(str, int)  # Передает путь к директории и ожидаемое количество изображений
    error_occurred = Signal(str)
    
    def __init__(
        self,
        network_pkl: str,
        seeds: List[int],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        translate: Tuple[float, float],
        rotate: float,
        class_idx: Optional[int]
    ) -> None:
        super().__init__()
        self.network_pkl: str = network_pkl
        self.seeds: List[int] = seeds
        self.truncation_psi: float = truncation_psi
        self.noise_mode: str = noise_mode
        self.outdir: str = outdir
        self.translate: Tuple[float, float] = translate
        self.rotate: float = rotate
        self.class_idx: Optional[int] = class_idx
        
    def run(self) -> None:
        try:
            self.log_message.emit(f'Загрузка сети из "{self.network_pkl}"...')
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
            with dnnlib.util.open_url(self.network_pkl) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
            
            os.makedirs(self.outdir, exist_ok=True)
            
            # Метки классов.
            label = torch.zeros([1, G.c_dim], device=device)
            if G.c_dim != 0:
                if self.class_idx is None:
                    self.error_occurred.emit('Необходимо указать метку класса (--class) при использовании условной сети')
                    return
                label[:, self.class_idx] = 1
            else:
                if self.class_idx is not None:
                    self.log_message.emit('Предупреждение: --class игнорируется при работе с безусловной сетью')
            
            # Генерация изображений.
            expected_count = len(self.seeds)
            generated_files = []
            
            for seed_idx, seed in enumerate(self.seeds):
                self.log_message.emit(f'Генерация изображения для seed {seed} ({seed_idx+1}/{expected_count})...')
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device).float()
                
                # Построить обратную матрицу поворота/переноса и передать генератору.
                if hasattr(G.synthesis, 'input'):
                    m = make_transform(self.translate, self.rotate)
                    m = np.linalg.inv(m)
                    G.synthesis.input.transform.copy_(torch.from_numpy(m))
                
                img = G(z, label, truncation_psi=self.truncation_psi, noise_mode=self.noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                filename = f'{self.outdir}/seed{seed:04d}.png'
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(filename)
                generated_files.append(filename)
                self.log_message.emit(f'Сгенерировано изображение: {filename}')
            
            # Убедимся, что ВСЕ файлы записаны на диск перед отправкой сигнала
            max_wait_time = 5.0  # Максимальное время ожидания в секундах
            wait_interval = 0.1  # Интервал проверки в секундах
            waited_time = 0.0
            
            self.log_message.emit('Проверка записи всех файлов на диск...')
            
            # Проверяем, что все ожидаемые файлы существуют на диске
            existing_files = []
            while waited_time < max_wait_time:
                existing_files = [f for f in generated_files if os.path.exists(f)]
                if len(existing_files) == expected_count:
                    # Проверяем, что файлы не пустые (размер > 0)
                    all_valid = all(os.path.getsize(f) > 0 for f in existing_files)
                    if all_valid:
                        break
                time.sleep(wait_interval)
                waited_time += wait_interval
            
            if len(existing_files) != expected_count:
                self.log_message.emit(f'Предупреждение: найдено {len(existing_files)} из {expected_count} файлов')
            
            self.log_message.emit(f'Генерация завершена! Всего изображений: {expected_count}')
            # Отправить сигнал только после генерации ВСЕХ изображений и проверки их наличия на диске
            self.generation_complete.emit(self.outdir, expected_count)
            
        except Exception as e:
            self.error_occurred.emit(f'Ошибка при генерации: {str(e)}')
