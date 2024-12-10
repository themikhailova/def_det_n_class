import sys
import os
import shutil
from PyQt5 import QtWidgets, QtCore
from design import Ui_MainWindow  # Импорт интерфейса из файла new.py


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()  # Инициализируем интерфейс
        self.ui.setupUi(self)  # Устанавливаем интерфейс в MainWindow

        # Храним текущую директорию и выбранный файл
        self.current_directory = ''
        self.selected_file = ''

        # Подключение сигналов к кнопкам
        self.ui.edit.clicked.connect(self.on_edit_clicked)
        self.ui.cancel.clicked.connect(self.on_cancel_clicked)
        self.ui.save.clicked.connect(self.on_save_clicked)
        self.ui.set_directory.clicked.connect(self.on_set_directory_clicked)
        self.ui.analysis.clicked.connect(self.on_analysis_clicked)
        self.ui.statistics.clicked.connect(self.on_statistics_clicked)
        self.ui.insert_template.clicked.connect(self.on_insert_template_clicked)
        self.ui.delete_template.clicked.connect(self.on_delete_template_clicked)

        # Подключаем событие клика по списку файлов
        self.ui.directory.clicked.connect(self.on_file_selected)

    def on_edit_clicked(self):
        print("Кнопка 'Изменить' нажата")

    def on_cancel_clicked(self):
        print("Кнопка 'Отменить' нажата")

    def on_save_clicked(self):
        """Сохранить выбранный файл в указанную директорию"""
        if not self.selected_file:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала выберите файл для сохранения.")
            return

        target_directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию для сохранения файла")
        if not target_directory:
            return  # Если пользователь закрыл диалог без выбора директории

        # Получаем имя файла из пути и сохраняем файл
        file_name = os.path.basename(self.selected_file)
        target_path = os.path.join(target_directory, file_name)

        try:
            shutil.copy(self.selected_file, target_path)
            QtWidgets.QMessageBox.information(self, "Успех",
                                              f"Файл '{file_name}' успешно сохранён в '{target_directory}'")
            print(f"Файл '{file_name}' сохранён в '{target_path}'")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")
            print(f"Ошибка сохранения файла: {e}")

    def on_set_directory_clicked(self):
        """Установить директорию для отображения изображений"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Указать директорию", "")
        if directory:
            self.current_directory = directory
            print(f"Выбрана директория: {directory}")
            self.load_files_from_directory(directory)

    def on_file_selected(self, index):
        """Когда пользователь выбирает файл из списка"""
        if self.current_directory:
            file_name = self.ui.directory.model().data(index)
            file_path = os.path.join(self.current_directory, file_name)
            self.selected_file = file_path
            print(f"Выбран файл: {file_path}")

    def load_files_from_directory(self, directory):
        """Загрузить изображения из указанной директории в ListView"""
        model = QtCore.QStringListModel()
        files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        model.setStringList(files)
        self.ui.directory.setModel(model)

    def on_analysis_clicked(self):
        print("Кнопка 'Провести анализ' нажата")

    def on_statistics_clicked(self):
        print("Кнопка 'Выгрузить статистику' нажата")

    def on_insert_template_clicked(self):
        print("Кнопка 'Добавить шаблон' нажата")

    def on_delete_template_clicked(self):
        print("Кнопка 'Удалить шаблон' нажата")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())