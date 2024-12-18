import sys
import re
import shutil
from pathlib import Path
from PyQt5 import QtWidgets, QtCore
from design import Ui_MainWindow  # Импорт интерфейса из файла new.py

#Константа для регулярного выражения фильтрации изображений
IMAGE_PATTERN = re.compile(r'.*\.(png|jpg|jpeg|gif|bmp)$', re.IGNORECASE)


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()  # Инициализируем интерфейс
        self.ui.setupUi(self)  # Устанавливаем интерфейс в MainWindow

        # Создаем модель для хранения списка файлов
        self.model = QtCore.QStringListModel()
        self.ui.directory.setModel(self.model)  # Назначаем модель виджету ListView

        # Подключение сигналов к кнопкам
        self.ui.edit.clicked.connect(self.on_edit_clicked)
        self.ui.cancel.clicked.connect(self.on_cancel_clicked)
        self.ui.set_directory.clicked.connect(self.on_set_directory_clicked)
        self.ui.analysis.clicked.connect(self.on_analysis_clicked)
        self.ui.statistics.clicked.connect(self.on_statistics_clicked)
        self.ui.insert_template.clicked.connect(self.on_insert_template_clicked)
        self.ui.delete_template.clicked.connect(self.on_delete_template_clicked)

        # Подключаем событие клика по списку файлов
        self.ui.directory.clicked.connect(self.on_file_selected)

        # Храним текущую директорию и выбранный файл
        self.current_directory = Path()
        self.selected_file = None  # Устанавливаем None, когда файл не выбран

        # Подключение сигнала к кнопке сохранения
        self.ui.save.clicked.connect(self.on_save_clicked)

    def on_edit_clicked(self):
        """Обработчик нажатия на кнопку 'Изменить'"""
        print("Кнопка 'Изменить' нажата")

    def on_cancel_clicked(self):
        """Обработчик нажатия на кнопку 'Отменить'"""
        print("Кнопка 'Отменить' нажата")

    def on_save_clicked(self):
        """Сохранить выбранный файл в указанную директорию"""
        if not self.selected_file or not self.selected_file.exists():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала выберите файл для сохранения.")
            return

        target_directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию для сохранения файла")
        if not target_directory:
            return  # Если пользователь закрыл диалог без выбора директории

        # Получаем имя файла из пути
        file_name = self.selected_file.name  # Получаем имя файла
        target_path = Path(target_directory) / file_name  # Конкатенация путей с помощью оператора /

        try:
            # Выполняем копирование файла
            shutil.copy(self.selected_file, target_path)
        except Exception as exc:
            # Выводим сообщение об ошибке
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка при копировании файла: {str(exc)}")
        else:
            # Выводим сообщение об успешном завершении копирования
            QtWidgets.QMessageBox.information(self, "Успех", f"Файл успешно скопирован в: {target_path}")

    def on_set_directory_clicked(self):
        """Установить директорию для отображения изображений"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Указать директорию", "")
        if directory:
            self.current_directory = Path(directory)
            print(f"Выбрана директория: {self.current_directory}")
            self.load_files_from_current_directory()

    def on_file_selected(self, index):
        """Когда пользователь выбирает файл из списка"""
        if self.current_directory:
            file_name = self.ui.directory.model().data(index, QtCore.Qt.DisplayRole)
            #Сохраняем полный путь к выбранному файлу
            self.selected_file = self.current_directory / file_name
            print(f"Выбран файл: {self.selected_file}")

    def load_files_from_current_directory(self):
        """Загрузить изображения из текущей директории self.current_directory в ListView"""
        if not self.current_directory or not self.current_directory.exists():
            print(f"Директория {self.current_directory} не найдена")
            return  # Если директория не установлена или не существует

        files = [file.name for file in self.current_directory.iterdir() if
                 file.is_file() and IMAGE_PATTERN.match(file.name)]
        print(f"Найдено файлов: {len(files)}")

        self.model.setStringList(files)  # Обновляем модель

    def on_analysis_clicked(self):
        """Обработчик нажатия на кнопку 'Провести анализ'"""
        print("Кнопка 'Провести анализ' нажата")

    def on_statistics_clicked(self):
        """Обработчик нажатия на кнопку 'Выгрузить статистику'"""
        print("Кнопка 'Выгрузить статистику' нажата")

    def on_insert_template_clicked(self):
        """Обработчик нажатия на кнопку 'Добавить шаблон'"""
        print("Кнопка 'Добавить шаблон' нажата")

    def on_delete_template_clicked(self):
        """Обработчик нажатия на кнопку 'Удалить шаблон'"""
        print("Кнопка 'Удалить шаблон' нажата")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())