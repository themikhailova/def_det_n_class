import sys
import re
import pandas as pd
import shutil
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from design import Ui_MainWindow, Ui_AddTemplateDialog
from PyQt5.QtCore import QAbstractTableModel, Qt

# Константа для регулярного выражения фильтрации изображений
IMAGE_PATTERN = re.compile(r'.*\.(png|jpg|jpeg|gif|bmp)$', re.IGNORECASE)


class PandasModel(QAbstractTableModel):
    """Модель для отображения DataFrame в QTableView с чекбоксами"""
    def __init__(self, data):
        super().__init__()
        self._data = data
        self.checked_row = -1  # Индекс выбранного чекбокса (один выбранный шаблон)

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1] + 1  # Добавление колонки для чекбоксов

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        # Колонка чекбоксов
        if index.column() == 0:
            if role == Qt.CheckStateRole:
                return Qt.Checked if index.row() == self.checked_row else Qt.Unchecked
            return None

        # Остальные данные
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column() - 1])
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if index.column() == 0 and role == Qt.CheckStateRole:
            self.checked_row = index.row()  # Обновляем индекс выбранного шаблона
            self.layoutChanged.emit()  # Обновляем отображение
            return True
        return False

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        return Qt.ItemIsEnabled

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section == 0:
                    return "Выбор"
                return self._data.columns[section - 1]
            if orientation == Qt.Vertical:
                return str(section + 1)
        return None

    def get_selected_template(self):
        """Получает данные выбранного шаблона"""
        if not hasattr(self, 'table_model') or self.table_model.checked_row == -1:
            return None
        return self.template_df.iloc[self.table_model.checked_row].to_dict()

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

        # Определим путь к файлу шаблонов в папке проекта
        project_dir = Path(__file__).parent  # Текущая директория проекта
        self.template_file_path = project_dir / "шаблоны.xlsx"

        # Загрузим существующую таблицу
        self.template_df = self.load_template_file()

        # Подключение модели к QTableView
        self.table_model = PandasModel(self.template_df)
        self.ui.template_2.setModel(self.table_model)

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
        print("Кнопка 'Изменить' нажата")

    def on_cancel_clicked(self):
        print("Кнопка 'Отменить' нажата")

    def on_save_clicked(self):
        if not self.selected_file or not self.selected_file.exists():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала выберите файл для сохранения.")
            return
        target_directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию для сохранения файла")
        if not target_directory:
            return
        file_name = self.selected_file.name
        target_path = Path(target_directory) / file_name
        try:
            shutil.copy(self.selected_file, target_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка при копировании файла: {str(exc)}")
        else:
            QtWidgets.QMessageBox.information(self, "Успех", f"Файл успешно скопирован в: {target_path}")

    def on_set_directory_clicked(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Указать директорию", "")
        if directory:
            self.current_directory = Path(directory)
            self.load_files_from_current_directory()

    def on_file_selected(self, index):
        if self.current_directory:
            file_name = self.ui.directory.model().data(index, QtCore.Qt.DisplayRole)
            self.selected_file = self.current_directory / file_name

    def load_files_from_current_directory(self):
        if not self.current_directory or not self.current_directory.exists():
            return
        files = [file.name for file in self.current_directory.iterdir() if
                 file.is_file() and IMAGE_PATTERN.match(file.name)]
        self.model.setStringList(files)

    def on_analysis_clicked(self):
        print("Кнопка 'Провести анализ' нажата")

    def on_statistics_clicked(self):
        """Обработчик нажатия на кнопку 'Выгрузить статистику'"""
        selected_template = self.get_selected_template()
        if selected_template:
            print(f"Выбран шаблон: {selected_template}")
    def on_insert_template_clicked(self):
        dialog = AddTemplateDialog()
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            template_data = dialog.get_template_data()
            self.add_template_to_table(template_data)

    def on_delete_template_clicked(self):
        """Обработчик нажатия на кнопку 'Удалить шаблон'"""
        selected_template = self.get_selected_template()
        if not selected_template:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите шаблон для удаления.")
            return

        # Подтверждение удаления
        confirm = QtWidgets.QMessageBox.question(
            self, "Удаление шаблона",
            f"Вы уверены, что хотите удалить шаблон:\n\nНаименование: {selected_template['Наименование']}\nКод изделия: {selected_template['Код изделия']}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        # Удаляем выбранный шаблон
        self.template_df = self.template_df.drop(self.table_model.checked_row).reset_index(drop=True)
        self.save_template_file()
        self.refresh_template_view()
        QtWidgets.QMessageBox.information(self, "Успех", "Шаблон успешно удален.")

    def get_selected_template(self):
        """Получает данные выбранного шаблона"""
        if not hasattr(self, 'table_model') or self.table_model.checked_row == -1:
            return None
        return self.template_df.iloc[self.table_model.checked_row].to_dict()

    def refresh_template_view(self):
        """Обновляет данные в QTableView"""
        self.table_model = PandasModel(self.template_df)
        self.ui.template_2.setModel(self.table_model)

    def add_template_to_table(self, template_data):
        required_columns = ["Наименование", "Код изделия", "Директория эталонов"]
        for col, key in zip(required_columns, ["name", "code", "directory"]):
            if not template_data.get(key) or not str(template_data.get(key)).strip():
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Поле '{col}' должно быть заполнено.")
                return
        new_row = pd.DataFrame([{
            "Наименование": template_data["name"],
            "Код изделия": template_data["code"],
            "Директория эталонов": template_data["directory"]
        }])
        self.template_df = pd.concat([self.template_df, new_row], ignore_index=True)
        self.template_df = self.template_df[required_columns]
        self.save_template_file()
        self.refresh_template_view()

    def load_template_file(self):
        if not self.template_file_path.exists():
            return pd.DataFrame(columns=["Наименование", "Код изделия", "Директория эталонов"])
        try:
            return pd.read_excel(self.template_file_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл шаблонов: {e}")
            return pd.DataFrame(columns=["Наименование", "Код изделия", "Директория эталонов"])

    def save_template_file(self):
        try:
            self.template_df.to_excel(self.template_file_path, index=False, engine="openpyxl")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить шаблоны: {e}")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())