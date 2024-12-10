from design import Ui_MainWindow  # Импорт сгенерированного файла design.py
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, QtGui
import os

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow() 
        self.ui.setupUi(self)

        # Подключение кнопок
        self.ui.pushButton_2.clicked.connect(self.on_save_clicked)
        self.ui.pushButton_3.clicked.connect(self.on_cancel_clicked)
        self.ui.pushButton.clicked.connect(self.on_change_clicked)
        self.ui.pushButton_6.clicked.connect(self.on_set_dir_clicked)
        self.ui.pushButton_7.clicked.connect(self.on_add_template_clicked)
        self.ui.pushButton_8.clicked.connect(self.on_delete_template_clicked)
        self.selected_directory = None  # Переменная для хранения выбранной директории

        # Подключение кнопок
        self.ui.pushButton_2.clicked.connect(self.on_save_image_clicked)  # Кнопка "Сохранить"
        self.ui.pushButton_6.clicked.connect(self.on_set_directory_clicked)  # Кнопка "Указать директорию"

        # Пример: добавляем изображение для сохранения
        self.image_to_save = None  # Переменная для изображения

        # Загружаем изображение для примера
        self.load_example_image()

    def load_example_image(self):
        """Загрузка изображения для примера (в реальном проекте замените на нужное)."""
        self.image_to_save = QtGui.QPixmap(300, 300)  # Создаем пустое изображение
        self.image_to_save.fill(QtGui.QColor("blue"))  # Заливаем цветом
        # Устанавливаем изображение в QLabel (допустим, у вас есть QLabel для отображения изображения)
        self.ui.label_2.setPixmap(self.image_to_save)

    def on_set_directory_clicked(self):
        """Выбор директории для сохранения."""
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию")
        if directory:
            self.selected_directory = directory
            print(f"Выбранная директория: {directory}")

    def on_save_image_clicked(self):
        """Сохранение изображения в выбранную директорию."""
        if not self.selected_directory:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала укажите директорию для сохранения!")
            return

        if self.image_to_save is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет изображения для сохранения!")
            return

        # Определяем путь для сохранения изображения
        file_path = os.path.join(self.selected_directory, "saved_image.png")

        try:
            # Сохраняем изображение
            self.image_to_save.save(file_path, "PNG")
            QtWidgets.QMessageBox.information(self, "Успех", f"Изображение сохранено по пути: {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить изображение: {e}")

    def on_save_clicked(self):
        print("Кнопка 'Сохранить' нажата")

    def on_cancel_clicked(self):
        print("Кнопка 'Отменить' нажата")

    def on_change_clicked(self):
        print("Кнопка 'Изменить' нажата")

    def on_set_dir_clicked(self):
        print("Кнопка 'Указать директорию' нажата")

    def on_add_template_clicked(self):
        print("Кнопка 'Добавить шаблон' нажата")

    def on_delete_template_clicked(self):
        print("Кнопка 'Удалить шаблон' нажата")



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())