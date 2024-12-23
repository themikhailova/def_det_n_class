import sys
import re
import pandas as pd
import shutil
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui  # Добавлен QtGui
from design import Ui_MainWindow, Ui_AddTemplateDialog

#Константа для регулярного выражения фильтрации изображений
IMAGE_PATTERN = re.compile(r'.*\.(png|jpg|jpeg|gif|bmp)$', re.IGNORECASE)

class AddTemplateDialog(QtWidgets.QDialog):
    """Диалоговое окно для добавления шаблона"""
    def __init__(self):
        super().__init__()
        self.ui = Ui_AddTemplateDialog()
        self.ui.setupUi(self)

        # Подключение кнопок
        self.ui.cancel_button.clicked.connect(self.reject)
        self.ui.save_button.clicked.connect(self.accept)
        self.ui.browseButton.clicked.connect(self.select_directory)

        # Инициализируем путь к изображению
        self.image_file = None

    def select_directory(self):
        """Открыть диалог выбора директории и обновить предпросмотр"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию")
        if directory:
            self.ui.directoryInput.setText(directory)
            self.update_preview(directory)

    def update_preview(self, directory):
        """Обновить предпросмотр изображения"""
        directory_path = Path(directory)
        image_files = [file for file in directory_path.iterdir() if file.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if image_files:
            # Сохраняем путь к первому изображению для адаптивного отображения
            self.image_file = image_files[0]
            self.render_preview()
        else:
            self.image_file = None
            self.ui.preview_label.setText("Нет доступных изображений")

    def render_preview(self):
        """Отобразить изображение адаптивно в зависимости от размеров метки"""
        if self.image_file:
            pixmap = QtGui.QPixmap(str(self.image_file))
            # Масштабируем изображение к размеру QLabel
            scaled_pixmap = pixmap.scaled(
                self.ui.preview_label.width(),
                self.ui.preview_label.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.ui.preview_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Перехват события изменения размера окна"""
        super().resizeEvent(event)
        self.render_preview()

    def get_template_data(self):
        """Возвращает данные из таблицы и директории"""
        name = self.ui.table.item(0, 0).text() if self.ui.table.item(0, 0) else ""
        code = self.ui.table.item(0, 1).text() if self.ui.table.item(0, 1) else ""
        directory = self.ui.directoryInput.text()
        return {"name": name, "code": code, "directory": directory}

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()  # Инициализируем интерфейс
        self.ui.setupUi(self)  # Устанавливаем интерфейс в MainWindow

        # Подключение сигналов к кнопкам
        self.ui.insert_template.clicked.connect(self.on_insert_template_clicked)

        # Путь к файлу шаблонов
        self.template_file_path = "/Users/valeriashchepilova/Desktop/шаблоны.xlsx"

        # Загрузим существующую таблицу
        self.template_df = self.load_template_file()

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
        dialog = AddTemplateDialog()
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            template_data = dialog.get_template_data()
            self.add_template_to_table(template_data)

    def on_delete_template_clicked(self):
        """Обработчик нажатия на кнопку 'Удалить шаблон'"""
        print("Кнопка 'Удалить шаблон' нажата")

    def load_template_file(self):
        """Загружает таблицу шаблонов или создает новую, если файл не найден"""
        try:
            return pd.read_excel(self.template_file_path)
        except FileNotFoundError:
            # Создаем пустую таблицу, если файл отсутствует
            return pd.DataFrame(columns=["Наименование", "Код изделия", "Директория эталонов"])

    def save_template_file(self):
        """Сохраняет таблицу шаблонов в файл"""
        try:
            self.template_df.to_excel(self.template_file_path, index=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить шаблоны: {e}")

    def add_template_to_table(self, template_data):
        """Добавляет новый шаблон в таблицу и сохраняет файл"""
        required_columns = ["Наименование", "Код изделия", "Директория эталонов"]

        # Проверяем, что все поля присутствуют и не являются пустыми
        for col, key in zip(required_columns, ["name", "code", "directory"]):
            if not template_data.get(key) or not str(template_data.get(key)).strip():
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Поле '{col}' должно быть заполнено.")
                return

        # Формируем новый DataFrame с учетом порядка столбцов
        new_row = pd.DataFrame([{
            "Наименование": template_data["name"],
            "Код изделия": template_data["code"],
            "Директория эталонов": template_data["directory"]
        }])

        # Добавляем новый шаблон в таблицу
        self.template_df = pd.concat([self.template_df, new_row], ignore_index=True)

        # Приводим таблицу к требуемой структуре
        self.template_df = self.template_df[required_columns]

        # Сохраняем таблицу
        self.save_template_file()

        QtWidgets.QMessageBox.information(self, "Успех", "Шаблон успешно добавлен и сохранен.")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())