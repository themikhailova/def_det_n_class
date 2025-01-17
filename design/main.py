import sys
import re
import pandas as pandas
import shutil
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from design import Ui_MainWindow, Ui_SelectTemplateDialog
from PyQt5.QtCore import QAbstractTableModel, Qt

# Константа для регулярного выражения фильтрации изображений
IMAGE_PATTERN = re.compile(r'.*\.(png|jpg|jpeg|gif|bmp)$', re.IGNORECASE)

class SelectTemplateDialog(QtWidgets.QDialog):
    """Диалоговое окно для добавления или редактирования шаблона"""

    def __init__(self, template_data=None):
        super().__init__()
        self.ui = Ui_SelectTemplateDialog()
        self.ui.setupUi(self)

        # Подключение кнопок
        self.ui.cancel_button.clicked.connect(self.reject)
        self.ui.save_button.clicked.connect(self.on_save_button_clicked)  # Меняем обработчик
        self.ui.browseButton.clicked.connect(self.select_directory)

        # Инициализируем путь к изображению
        self.image_file = None

        # Если переданы данные шаблона, заполняем поля
        if template_data:
            self.prepopulate_fields(template_data)

    def on_save_button_clicked(self):
        """Обработчик нажатия кнопки 'Сохранить'"""
        # Принудительно завершить редактирование текущей ячейки таблицы
        self.ui.table.clearFocus()

        # Константы для индексов ячеек
        NAME_ROW, NAME_COL = 0, 0
        CODE_ROW, CODE_COL = 0, 1

        # Извлекаем элементы из таблицы
        name_item = self.ui.table.item(NAME_ROW, NAME_COL)
        code_item = self.ui.table.item(CODE_ROW, CODE_COL)

        # Извлекаем значения с проверкой
        name = name_item.text() if name_item else ""
        code = code_item.text() if code_item else ""
        directory = self.ui.directoryInput.text().strip() if self.ui.directoryInput.text() else None

        # Проверяем заполненность всех полей
        if not name or not code or not directory:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Все поля должны быть заполнены!")
            return

        # Если всё заполнено, принимаем диалог
        self.accept()

    def prepopulate_fields(self, template_data):
        """Предзаполнение полей на основе переданного шаблона"""
        # Предзаполнение таблицы
        name_item = QtWidgets.QTableWidgetItem(template_data.get("Наименование", ""))
        code_item = QtWidgets.QTableWidgetItem(template_data.get("Код изделия", ""))
        self.ui.table.setItem(0, 0, name_item)
        self.ui.table.setItem(0, 1, code_item)

        # Установка директории и предпросмотра
        directory = template_data.get("Директория эталонов", "")
        self.ui.directoryInput.setText(directory)
        if directory:
            self.update_preview(directory)

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
            self.image_file = image_files[0]
            self.render_preview()
        else:
            self.image_file = None
            self.ui.preview_label.setText("Нет доступных изображений")

    def render_preview(self):
        """Отобразить изображение адаптивно в зависимости от размеров метки"""
        if self.image_file:
            pixmap = QtGui.QPixmap(str(self.image_file))
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
        """Возвращает данные из таблицы и директории (всегда корректный словарь)"""
        # Константы для индексов ячеек
        NAME_ROW, NAME_COL = 0, 0
        CODE_ROW, CODE_COL = 0, 1

        # Извлекаем элементы из таблицы
        name_item = self.ui.table.item(NAME_ROW, NAME_COL)
        code_item = self.ui.table.item(CODE_ROW, CODE_COL)

        # Извлекаем значения
        name = name_item.text() if name_item else ""
        code = code_item.text() if code_item else ""
        directory = self.ui.directoryInput.text().strip() if self.ui.directoryInput.text() else ""

        return {"Наименование": name, "Код изделия": code, "Директория эталонов": directory}


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Определяем файл шаблонов
        self.project_dir = Path(__file__).parent
        self.template_file_path = self.project_dir / "шаблоны.xlsx"

        # Загружаем данные из Excel файла
        self.template_df = self.load_template_file()

        # Подключение модели
        self.table_model = PandasModel(self.template_df)
        self.ui.template_2.setModel(self.table_model)

        # Подключение сигналов
        self.ui.template_2.doubleClicked.connect(self.on_template_double_clicked)
        self.ui.insert_template.clicked.connect(self.on_insert_template_clicked)
        self.ui.delete_template.clicked.connect(self.on_delete_template_clicked)
        self.ui.set_directory.clicked.connect(self.on_set_directory_clicked)

    def on_template_double_clicked(self, index):
        """Открывает диалоговое окно для редактирования шаблона при двойном клике"""
        if index.row() < 0:
            return  # Если индекс невалидный

        # Получаем данные выбранного шаблона
        selected_template = self.template_df.iloc[index.row()].to_dict()

        # Проверяем наличие всех нужных ключей
        required_keys = ["Наименование", "Код изделия", "Директория эталонов"]
        for key in required_keys:
            if key not in selected_template:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Отсутствует поле: {key}")
                return

        # Открываем диалоговое окно редактирования
        dialog = SelectTemplateDialog(selected_template)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Получаем обновленные данные из диалога
            updated_data = dialog.get_template_data()

            # Обновляем данные в DataFrame
            for key, value in updated_data.items():
                self.template_df.at[index.row(), key] = value

            # Сохраняем изменения и обновляем отображение
            self.save_template_file()
            self.refresh_template_view()

    def on_insert_template_clicked(self):
        """Добавление нового шаблона"""
        dialog = SelectTemplateDialog()  # Мы открываем диалог для нового шаблона
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            template_data = dialog.get_template_data()  # Получаем данные из диалога
            # Нет необходимости в проверке, так как данные всегда валидны
            self.add_template_to_table(template_data)

    def on_delete_template_clicked(self):
        """Удаление выбранного шаблона"""
        if self.table_model.checked_row == -1:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите шаблон для удаления.")
            return

        selected_template = self.template_df.iloc[self.table_model.checked_row].to_dict()
        template_name = selected_template.get("Наименование", "без имени")
        template_code = selected_template.get("Код изделия", "без кода")

        # Подтверждение удаления
        confirm = QtWidgets.QMessageBox.question(
            self, "Удаление шаблона",
            f"Вы уверены, что хотите удалить шаблон?\n\nНаименование: {template_name}\nКод изделия: {template_code}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        # Удаляем выбранный шаблон
        self.template_df = self.template_df.drop(self.table_model.checked_row).reset_index(drop=True)
        self.save_template_file()
        self.refresh_template_view()
        QtWidgets.QMessageBox.information(self, "Успех", "Шаблон успешно удален.")

    def on_set_directory_clicked(self):
        """Установить директорию для отображения изображений"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Указать директорию", "")
        if directory:
            self.current_directory = Path(directory)
            self.load_images_to_view()

    def load_images_to_view(self):
        """Загружает изображения из текущей директории и отображает их в QListWidget"""
        if not self.current_directory or not self.current_directory.exists():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Директория не выбрана или не существует.")
            return

        image_files = [file for file in self.current_directory.iterdir() if IMAGE_PATTERN.match(file.name)]

        # Очищаем текущий виджет, чтобы избежать старых данных
        self.ui.image_list.clear()

        if not image_files:
            # Показываем только сообщение, без добавления текста в QListWidget
            QtWidgets.QMessageBox.information(self, "Информация", "В выбранной директории нет изображений.")
            return

        # Если изображения есть, то добавляем их в список
        for image_file in image_files:
            item = QtWidgets.QListWidgetItem()
            pixmap = QtGui.QPixmap(str(image_file))

            # Масштабируем изображение к ширине QListWidget
            widget_width = self.ui.image_list.width() - 10  # Учитываем отступы
            scaled_pixmap = pixmap.scaled(
                widget_width,
                widget_width,  # Используем квадратное изображение
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )

            item.setIcon(QtGui.QIcon(scaled_pixmap))
            item.setText(image_file.name)
            item.setToolTip(str(image_file))
            self.ui.image_list.addItem(item)

        # Устанавливаем размер элементов в QListWidget
        self.ui.image_list.setIconSize(QtCore.QSize(widget_width, widget_width))

    def add_template_to_table(self, template_data):
        """Добавить шаблон в таблицу"""
        new_row = pandas.DataFrame([{
            "Наименование": template_data["Наименование"],  # Правильный ключ
            "Код изделия": template_data["Код изделия"],  # Правильный ключ
            "Директория эталонов": template_data["Директория эталонов"]  # Правильный ключ
        }])

        # Добавляем новый шаблон в таблицу без дополнительной фильтрации колонок
        self.template_df = pandas.concat([self.template_df, new_row], ignore_index=True)

        # Сохраняем и обновляем отображение
        self.save_template_file()
        self.refresh_template_view()

    def refresh_template_view(self):
        """Обновить отображение таблицы шаблонов"""
        self.table_model.update_data(self.template_df)  # Обновляем данные модели

    def save_template_file(self):
        """Сохранение шаблонов в файл"""
        self.template_df.to_excel(self.template_file_path, index=False, engine="openpyxl")

    def load_template_file(self):
        """Загрузить таблицу шаблонов из файла"""
        if not self.template_file_path.exists():
            return pandas.DataFrame(columns=["Наименование", "Код изделия", "Директория эталонов"])

        df = pandas.read_excel(self.template_file_path)
        # Оставляем только нужные колонки
        return df[["Наименование", "Код изделия", "Директория эталонов"]]


class PandasModel(QAbstractTableModel):
    """Модель для отображения DataFrame в QTableView с чекбоксами"""
    def __init__(self, data):
        super().__init__()
        self._data = data
        self.checked_row = -1  # Индекс выбранного чекбокса

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

    def update_data(self, new_data):
        """Обновить данные модели"""
        self._data = new_data
        self.layoutChanged.emit()  # Сообщить представлению об изменении данных

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())