# 3rdparty
from PySide6.QtWidgets import QApplication

# project
from .main_window import ImageGeneratorGUI


def main():
    """Главная функция запуска приложения."""
    app = QApplication([])
    window = ImageGeneratorGUI()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
