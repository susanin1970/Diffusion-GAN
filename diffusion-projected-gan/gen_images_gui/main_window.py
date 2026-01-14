# python
import os
import glob

# 3rdparty
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QFileDialog,
    QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

# project
from .threads import GenerationWorker
from .utils import parse_range, parse_vec2


class ImageGeneratorGUI(QMainWindow):
    """Главное окно GUI приложения"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Генератор изображений - Diffusion Projected GAN')
        self.setGeometry(100, 100, 1000, 800)
        
        # Переменные для управления изображениями
        self.current_image_index = 0
        self.image_files = []
        self.outdir = ""
        
        # Рабочий поток генерации
        self.worker = None
        
        self.init_ui()
        
    def init_ui(self):
        """Инициализация интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # === 1. Верхняя часть: поля ввода параметров ===
        params_layout = QVBoxLayout()
        
        # Определяем минимальную ширину для всех лейблов для выравнивания
        label_min_width = 200
        
        # Сеть
        network_layout = QHBoxLayout()
        network_label = QLabel('Network (--network):')
        network_label.setMinimumWidth(label_min_width)
        network_layout.addWidget(network_label)
        self.network_input = QLineEdit()
        self.network_input.setPlaceholderText('Путь к файлу сети или URL')
        network_layout.addWidget(self.network_input, stretch=1)
        browse_network_btn = QPushButton('Обзор...')
        browse_network_btn.clicked.connect(self.browse_network_file)
        network_layout.addWidget(browse_network_btn)
        params_layout.addLayout(network_layout)
        
        # Семена (seeds)
        seeds_layout = QHBoxLayout()
        seeds_label = QLabel('Seeds (--seeds):')
        seeds_label.setMinimumWidth(label_min_width)
        seeds_layout.addWidget(seeds_label)
        self.seeds_input = QLineEdit()
        self.seeds_input.setPlaceholderText('Например: 0,1,4-6')
        seeds_layout.addWidget(self.seeds_input, stretch=1)
        params_layout.addLayout(seeds_layout)
        
        # Truncation Psi
        trunc_layout = QHBoxLayout()
        trunc_label = QLabel('Truncation Psi (--trunc):')
        trunc_label.setMinimumWidth(label_min_width)
        trunc_layout.addWidget(trunc_label)
        self.trunc_input = QLineEdit('1')
        self.trunc_input.setPlaceholderText('1')
        trunc_layout.addWidget(self.trunc_input, stretch=1)
        params_layout.addLayout(trunc_layout)
        
        # Класс
        class_layout = QHBoxLayout()
        class_label = QLabel('Class (--class):')
        class_label.setMinimumWidth(label_min_width)
        class_layout.addWidget(class_label)
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText('Опционально: метка класса')
        class_layout.addWidget(self.class_input, stretch=1)
        params_layout.addLayout(class_layout)
        
        # Режим шума
        noise_layout = QHBoxLayout()
        noise_label = QLabel('Noise Mode (--noise-mode):')
        noise_label.setMinimumWidth(label_min_width)
        noise_layout.addWidget(noise_label)
        self.noise_mode_combo = QComboBox()
        self.noise_mode_combo.addItems(['const', 'random', 'none'])
        self.noise_mode_combo.setCurrentText('const')
        noise_layout.addWidget(self.noise_mode_combo, stretch=1)
        params_layout.addLayout(noise_layout)
        
        # Перенос
        translate_layout = QHBoxLayout()
        translate_label = QLabel('Translate (--translate):')
        translate_label.setMinimumWidth(label_min_width)
        translate_layout.addWidget(translate_label)
        self.translate_input = QLineEdit('0,0')
        self.translate_input.setPlaceholderText('0,0')
        translate_layout.addWidget(self.translate_input, stretch=1)
        params_layout.addLayout(translate_layout)
        
        # Поворот
        rotate_layout = QHBoxLayout()
        rotate_label = QLabel('Rotate (--rotate):')
        rotate_label.setMinimumWidth(label_min_width)
        rotate_layout.addWidget(rotate_label)
        self.rotate_input = QLineEdit('0')
        self.rotate_input.setPlaceholderText('0')
        rotate_layout.addWidget(self.rotate_input, stretch=1)
        params_layout.addLayout(rotate_layout)
        
        # Выходная директория
        outdir_layout = QHBoxLayout()
        outdir_label = QLabel('Output Directory (--outdir):')
        outdir_label.setMinimumWidth(label_min_width)
        outdir_layout.addWidget(outdir_label)
        self.outdir_input = QLineEdit('out')
        self.outdir_input.setPlaceholderText('out')
        outdir_layout.addWidget(self.outdir_input, stretch=1)
        browse_outdir_btn = QPushButton('Обзор...')
        browse_outdir_btn.clicked.connect(self.browse_output_directory)
        outdir_layout.addWidget(browse_outdir_btn)
        params_layout.addLayout(outdir_layout)
        
        # Кнопка генерации
        self.generate_btn = QPushButton('Сгенерировать изображения')
        self.generate_btn.clicked.connect(self.start_generation)
        params_layout.addWidget(self.generate_btn)
        
        main_layout.addLayout(params_layout)
        
        # === 2. Окно предпросмотра изображений ===
        preview_label = QLabel('Предпросмотр изображений')
        preview_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(preview_label)
        
        self.preview_label = QLabel('Изображения будут отображены здесь после генерации')
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(400)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.preview_label.setScaledContents(False)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.preview_label)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        main_layout.addWidget(scroll_area)
        
        # === 3. Кнопки перелистывания ===
        navigation_layout = QHBoxLayout()
        self.prev_btn = QPushButton('◀ Предыдущее')
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.prev_btn.setEnabled(False)
        navigation_layout.addWidget(self.prev_btn)
        
        self.image_counter_label = QLabel('0 / 0')
        self.image_counter_label.setAlignment(Qt.AlignCenter)
        navigation_layout.addWidget(self.image_counter_label)
        
        self.next_btn = QPushButton('Следующее ▶')
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)
        navigation_layout.addWidget(self.next_btn)
        
        main_layout.addLayout(navigation_layout)
        
        # === 4. Окно логирования ===
        log_label = QLabel('Логирование:')
        main_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        main_layout.addWidget(self.log_text)
        
        self.log_message('Готов к генерации изображений')
        
    def browse_network_file(self):
        """Открыть диалог выбора файла сети"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите файл сети', '', 'Pickle Files (*.pkl);;All Files (*)'
        )
        if file_path:
            self.network_input.setText(file_path)
    
    def browse_output_directory(self):
        """Открыть диалог выбора директории для сохранения"""
        dir_path = QFileDialog.getExistingDirectory(self, 'Выберите директорию для сохранения')
        if dir_path:
            self.outdir_input.setText(dir_path)
    
    def log_message(self, message: str):
        """Добавить сообщение в лог"""
        self.log_text.append(message)
        # Автопрокрутка вниз
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def validate_inputs(self) -> tuple:
        """Проверка и парсинг входных данных"""
        try:
            network_pkl = self.network_input.text().strip()
            if not network_pkl:
                raise ValueError('Необходимо указать путь к сети (--network)')
            
            seeds_str = self.seeds_input.text().strip()
            if not seeds_str:
                raise ValueError('Необходимо указать seeds (--seeds)')
            seeds = parse_range(seeds_str)
            
            trunc_str = self.trunc_input.text().strip()
            truncation_psi = float(trunc_str) if trunc_str else 1.0
            
            class_str = self.class_input.text().strip()
            class_idx = int(class_str) if class_str else None
            
            noise_mode = self.noise_mode_combo.currentText()
            
            translate_str = self.translate_input.text().strip()
            translate = parse_vec2(translate_str) if translate_str else (0.0, 0.0)
            
            rotate_str = self.rotate_input.text().strip()
            rotate = float(rotate_str) if rotate_str else 0.0
            
            outdir = self.outdir_input.text().strip()
            if not outdir:
                raise ValueError('Необходимо указать директорию для сохранения (--outdir)')
            
            return (network_pkl, seeds, truncation_psi, noise_mode, outdir, translate, rotate, class_idx)
            
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка валидации', f'Ошибка в параметрах: {str(e)}')
            return None
    
    def start_generation(self):
        """Запуск генерации изображений"""
        params = self.validate_inputs()
        if params is None:
            return
        
        network_pkl, seeds, truncation_psi, noise_mode, outdir, translate, rotate, class_idx = params
        
        # Отключить кнопку генерации во время работы
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText('Генерация...')
        
        # Очистить предыдущие изображения и скрыть их до завершения генерации
        self.current_image_index = 0
        self.image_files = []
        self.outdir = outdir
        self.preview_label.setText(f'Генерация изображений... (будет сгенерировано: {len(seeds)})')
        self.preview_label.setPixmap(QPixmap())  # Очистить предыдущее изображение
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.image_counter_label.setText('0 / 0')
        
        # Создать и запустить рабочий поток
        self.worker = GenerationWorker(
            network_pkl, seeds, truncation_psi, noise_mode, 
            outdir, translate, rotate, class_idx
        )
        self.worker.log_message.connect(self.log_message)
        self.worker.generation_complete.connect(self.on_generation_complete)
        self.worker.error_occurred.connect(self.on_generation_error)
        self.worker.start()
    
    def on_generation_complete(self, outdir: str, expected_count: int):
        """Обработка завершения генерации"""
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText('Сгенерировать изображения')
        
        # Обновить текст предпросмотра
        self.preview_label.setText(f'Загрузка изображений... (ожидается: {expected_count})')
        self.preview_label.setPixmap(QPixmap())  # Очистить предыдущее изображение
        
        # Загрузить список изображений только после завершения генерации всех изображений
        self.load_images(outdir)
        
        # Проверить, что найдено ожидаемое количество изображений
        if len(self.image_files) == expected_count:
            self.log_message(f'Все изображения загружены: {len(self.image_files)} из {expected_count}')
            # Показать первое изображение только после загрузки ВСЕХ изображений
            self.show_image(0)
        elif len(self.image_files) > 0:
            self.log_message(f'Загружено {len(self.image_files)} из {expected_count} изображений')
            # Показать первое изображение, даже если не все найдены
            self.show_image(0)
        else:
            self.preview_label.setText('Изображения не найдены')
            self.log_message('Ошибка: изображения не найдены в выходной директории')
    
    def on_generation_error(self, error_message: str):
        """Обработка ошибки генерации"""
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText('Сгенерировать изображения')
        QMessageBox.critical(self, 'Ошибка генерации', error_message)
    
    def load_images(self, directory: str):
        """Загрузить список изображений из директории"""
        self.image_files = sorted(glob.glob(os.path.join(directory, '*.png')))
        self.image_files.extend(sorted(glob.glob(os.path.join(directory, '*.jpg'))))
        self.image_files.extend(sorted(glob.glob(os.path.join(directory, '*.jpeg'))))
        
        if self.image_files:
            self.log_message(f'Загружено изображений: {len(self.image_files)}')
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
        else:
            self.log_message('Изображения не найдены в директории')
    
    def show_image(self, index: int):
        """Показать изображение по индексу"""
        if not self.image_files or index < 0 or index >= len(self.image_files):
            return
        
        self.current_image_index = index
        image_path = self.image_files[index]
        
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.log_message(f'Не удалось загрузить изображение: {image_path}')
            return
        
        # Масштабировать изображение для предпросмотра, сохраняя пропорции
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
        
        # Обновить счетчик
        self.image_counter_label.setText(f'{index + 1} / {len(self.image_files)}')
        
        filename = os.path.basename(image_path)
        self.log_message(f'Отображено изображение: {filename}')
    
    def show_previous_image(self):
        """Показать предыдущее изображение"""
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)
    
    def show_next_image(self):
        """Показать следующее изображение"""
        if self.current_image_index < len(self.image_files) - 1:
            self.show_image(self.current_image_index + 1)
    
    def resizeEvent(self, event):
        """Обработка изменения размера окна"""
        super().resizeEvent(event)
        # Обновить изображение при изменении размера окна
        if self.image_files and 0 <= self.current_image_index < len(self.image_files):
            self.show_image(self.current_image_index)
