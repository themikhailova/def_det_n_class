import sys, os
import re
import pandas as pandas
import cv2
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QAbstractTableModel, Qt, QCoreApplication
from PIL import Image
from PyQt5.QtGui import QPixmap
import ast  # Для преобразования строки в кортеж


sys.path.append(os.path.abspath('./designUI'))
from designUI.design import Ui_MainWindow, Ui_SelectTemplateDialog

sys.path.append(os.path.abspath('./detect'))
from detect.character import set_reference_path
from detect.classif import retrain_model

# Укажите путь к папке plugins
os.environ['QT_PLUGIN_PATH'] = r"./venv/Lib/site-packages/PyQt5/Qt5/plugins"
QCoreApplication.addLibraryPath(r"./venv/Lib/site-packages/PyQt5/Qt5/plugins")

# Константа для регулярного выражения фильтрации изображений
IMAGE_PATTERN = re.compile(r'.*\.(png|jpg|jpeg|gif|bmp)$', re.IGNORECASE)

# Константа для отступов в `QListWidget`
LIST_WIDGET_PADDING = 10  # Отступы внутри QListWidget

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

class ImageNavigator:
    def __init__(self, ui, base_image_path):
        self.ui = ui
        self.base_image_path = base_image_path
        self.excel_path = "./anomalies.xlsx"
        self.current_index = 0
        self.df = None

        # self.load_excel_data()

        self.ui.btn_left.clicked.connect(self.show_previous_image)
        self.ui.btn_right.clicked.connect(self.show_next_image)

        # self.load_current_image()

    def load_excel_data(self):
        """Загружает данные из Excel."""
        if not os.path.exists(self.excel_path):
            # QtWidgets.QMessageBox.critical(self, "Ошибка", f"Файл  не найден!")
            return
        # try:
        self.df = pandas.read_excel(self.excel_path)
        if self.df.empty:
            print("Файл Excel пуст.")
                # QtWidgets.QMessageBox.information(self, "Информация", "Файл Excel пуст.")
        # except Exception as e:
        #     QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки Excel: {e}")

    def load_current_image(self):
        """Загружает изображение и накладывает bounding box."""
        if not os.path.exists(self.excel_path):
            # QtWidgets.QMessageBox.critical(self, "Ошибка", f"Файл  не найден!")
            return
        # try:
        self.df = pandas.read_excel(self.excel_path)
        if self.df is None or self.df.empty:
            print("⛔️ Данные из Excel не загружены или пусты.")
            return

        if not os.path.exists(self.base_image_path):
            print(f"❌ Файл {self.base_image_path} не найден.")
            return

        # Получаем имя файла для текущей аномалии
        try:
            current_filename = self.df.iloc[self.current_index]["anomaly_filename"]
        except KeyError:
            print("❌ Столбец 'anomaly_filename' не найден в Excel.")
            return

        # Загружаем изображение
        file_path = os.path.join(self.base_image_path, current_filename)
        if not os.path.exists(file_path):
            print(f"❌ Файл {file_path} не найден.")
            return

        base_pixmap = QtGui.QPixmap(file_path)
        if base_pixmap.isNull():
            print(f"❌ Ошибка загрузки изображения: {file_path}")
            return

        # Масштабируем изображение
        view_width = self.ui.defect.width()
        original_width = base_pixmap.width()
        original_height = base_pixmap.height()

        scaled_pixmap = base_pixmap.scaled(view_width - 20, 300, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        scale_x = scaled_pixmap.width() / original_width
        scale_y = scaled_pixmap.height() / original_height

        painter = QtGui.QPainter(scaled_pixmap)
        row = self.df.iloc[self.current_index]
        x, y, w, h = row["bounding_rect_x"], row["bounding_rect_y"], row["bounding_rect_w"], row["bounding_rect_h"]
        
        # Получаем цвет аномалии
        try:
            colour = ast.literal_eval(row["anomaly_colour"])
            if not (isinstance(colour, tuple) and len(colour) == 3):
                raise ValueError
        except (ValueError, SyntaxError):
            colour = (0, 0, 255)  # по умолчанию синий

        # Масштабируем координаты и размер
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)
        w_scaled = int(w * scale_x)
        h_scaled = int(h * scale_y)

        # Рисуем bounding box
        painter.drawRect(x_scaled, y_scaled, w_scaled, h_scaled)

        pen = QtGui.QPen(QtGui.QColor(*colour))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(x_scaled, y_scaled, w_scaled, h_scaled)
        painter.end()

        self.ui.defect.setPixmap(scaled_pixmap)

    def show_previous_image(self):
        """Переключение на предыдущую строку в Excel."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
        else:
            print("🔴 Достигнуто начало данных в Excel")

    def show_next_image(self):
        """Переключение на следующую строку в Excel."""
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
            self.load_current_image()
        else:
            print("🔴 Достигнут конец данных в Excel")

    def reload_data(self):
        """Перезагружаем данные из Excel и обновляем изображение."""
        print("🔄 Перезагружаем данные из Excel...")
        self.load_excel_data()  # Загружаем обновленные данные
        self.load_current_image()  # Обновляем текущее изображение

    def update_anomaly(self, new_anomaly, new_colour):
        """Обновление аномалии в Excel для текущего изображения."""
        if self.df is None or self.df.empty:
            print("⛔️ Данные не загружены.")
            return

        # Получаем текущий файл
        try:
            current_filename = self.df.iloc[self.current_index]["anomaly_filename"]
        except KeyError:
            print("❌ Столбец 'anomaly_filename' не найден в Excel.")
            return

        # Обновляем значения аномалии и цвета
        self.df.loc[self.df["anomaly_filename"] == current_filename, "Y"] = new_anomaly
        self.df.loc[self.df["anomaly_filename"] == current_filename, "anomaly_colour"] = str(new_colour)
        self.reload_data()

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.selected_directory = None  # Переменная для хранения пути к директории
        self.export_directory= None
        self.output_folder = "C:/output_folder/"
        
        # Передаём этот путь в ImageNavigator
        self.image_navigator = ImageNavigator(self.ui, self.output_folder)

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
        # self.ui.analysis.clicked.connect(self.on_run_analysis_clicked)
        self.ui.statistics.clicked.connect(self.on_export_clicked)
        self.ui.edit.clicked.connect(self.on_edit_clicked)
        self.ui.save.clicked.connect(self.on_save_clicked)
        # Папка с результатами анализа
        
        self.excel_path = "./anomalies.xlsx"
        self.base_image_path = "./input_img.png"
        self.anomalies_types = "./types.xlsx"
        # Таймер для проверки новых файлов
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_image_display)

        # Подключаем кнопку "Провести анализ"
        self.ui.analysis.clicked.connect(self.on_run_analysis_clicked)

        # Список отображённых файлов, чтобы не загружать их повторно
        self.processed_files = set()

    def resizeEvent(self, event):
        """Обновляет масштаб изображения при изменении размера окна, но не увеличивает бесконечно"""
        super().resizeEvent(event)

        if hasattr(self.image_navigator, "image_files") and self.image_navigator.image_files:
            image_path = os.path.join(
                self.image_navigator.output_folder,
                self.image_navigator.image_files[self.image_navigator.current_index]
            )
            pixmap = QPixmap(image_path)

            label_width = self.ui.defect.width()
            label_height = self.ui.defect.height()

            scaled_pixmap = pixmap.scaled(
                label_width,
                label_height,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )

            self.ui.defect.setPixmap(scaled_pixmap)
            self.ui.defect.setScaledContents(False)  # Важно отключить бесконечное масштабирование

    def is_valid_file(self, file_path):
        """Проверка, является ли файл изображением или моделью .obj/.stl"""
        valid_image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        valid_model_extensions = {'.obj', '.stl'}
        
        file_extension = Path(file_path).suffix.lower()
        
        return file_extension in valid_image_extensions or file_extension in valid_model_extensions

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
            # Проверка содержимого директории для нового шаблона
            directory_path = updated_data.get("Директория эталонов")
            if not os.path.isdir(directory_path):
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Указанная директория не существует.")
                return

            # Получаем все файлы в директории и проверяем их расширения
            valid_files_found = False
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path) and self.is_valid_file(file_path):
                    valid_files_found = True
                    break

            if not valid_files_found:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "В указанной директории нет поддерживаемых файлов (изображений или моделей .obj/.stl).")
                return
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
            # Проверка содержимого директории для нового шаблона
            directory_path = template_data.get("Директория эталонов")
            if not os.path.isdir(directory_path):
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Указанная директория не существует.")
                return

            # Получаем все файлы в директории и проверяем их расширения
            valid_files_found = False
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path) and self.is_valid_file(file_path):
                    valid_files_found = True
                    break

            if not valid_files_found:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "В указанной директории нет поддерживаемых файлов (изображений или моделей .obj/.stl).")
                return
            
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

    def on_save_clicked(self):
        new_data_path = self.excel_path
        if not os.path.exists(new_data_path):
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Файл {new_data_path} не найден!")
            return
        try:
            retrain_model(new_data_path)
            QtWidgets.QMessageBox.information(self, "Успех", f"Модель дообучена на новых данных")
        except PermissionError as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Открыт файл.")
        

    def on_edit_clicked(self):
        """Обработчик нажатия на кнопку 'Изменить'"""

        # Загружаем Excel с типами аномалий
        try:
            types_df = pandas.read_excel(self.anomalies_types)
        except FileNotFoundError:
            types_df = pandas.DataFrame(columns=["anomaly_type", "anomaly_colour"])  
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить список аномалий: {str(e)}")
            return

        # Преобразуем таблицу в словарь {аномалия: цвет}
        anomaly_colors = {
            row["anomaly_type"]: ast.literal_eval(row["anomaly_colour"])
            for _, row in types_df.iterrows()
        }

        # Добавляем "new anomaly" в список категорий
        categories = list(anomaly_colors.keys()) + ["new anomaly"]

        # Выбор категории аномалии
        new_anomaly_name, ok = QtWidgets.QInputDialog.getItem(
            self, "Выбор аномалии", "Выберите категорию аномалии:", categories, 0, False
        )
        if not ok:
            return  # Отмена выбора

        # Если выбрана новая аномалия, запрашиваем имя
        if new_anomaly_name == 'new anomaly':
            new_anomaly_name, ok = QtWidgets.QInputDialog.getText(
                self, "Новая аномалия", "Введите название новой аномалии:"
            )
            if not ok or not new_anomaly_name.strip():
                return  # Отмена ввода

            # Открываем диалог выбора цвета
            color = QtWidgets.QColorDialog.getColor()

            if not color.isValid():
                return  # Отмена выбора цвета

            # Получаем цвет в формате (R, G, B)
            new_colour = (color.red(), color.green(), color.blue())

            # Добавляем новую аномалию в types.xlsx
            new_anomaly_row = pandas.DataFrame({"anomaly_type": [new_anomaly_name], "anomaly_colour": [str(new_colour)]})
            types_df = pandas.concat([types_df, new_anomaly_row], ignore_index=True)
            try:
                types_df.to_excel(self.anomalies_types, index=False)
            except PermissionError as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Открыт файл.")
        else:
            # Берем цвет из загруженного списка
            new_colour = anomaly_colors.get(new_anomaly_name, (128, 128, 128))  # Серый по умолчанию

        # Проверяем, не пустая ли директория
        if not os.listdir(self.output_folder):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Директория пуста. Невозможно выполнить изменение.")
            return

        if not os.path.exists(self.excel_path):
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Файл {self.excel_path} не найден!")
            return

        try:
            # Загружаем Excel с файлами
            df = pandas.read_excel(self.excel_path)

            # Получаем имя файла для текущего изображения
            current_filename = self.image_navigator.df.iloc[self.image_navigator.current_index]["anomaly_filename"]
            old_filepath = os.path.join(self.output_folder, current_filename)

            match = re.match(r'C:/output_folder/(\d+)_anomaly_(\w+)\.png', current_filename)
            if not match:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Неверный формат имени файла.")
                return

            # Обновляем Excel: заменяем аномалию и цвет для текущего изображения
            df.loc[df["anomaly_filename"] == old_filepath, "Y"] = new_anomaly_name
            df.loc[df["anomaly_filename"] == old_filepath, "anomaly_colour"] = str(new_colour)  # Сохраняем в строковом формате

            # Сохраняем изменения
            df.to_excel(self.excel_path, index=False)

            print(f"Файл и таблица обновлены: {old_filepath}")
            QtWidgets.QMessageBox.information(self, "Успех", f"Файл обновлен в таблице.")

            # Используем метод update_anomaly из ImageNavigator
            self.image_navigator.update_anomaly(new_anomaly_name, new_colour)

            # Перезагружаем данные и обновляем изображение
            self.image_navigator.reload_data()

        except PermissionError as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Открыт файл.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Произошла ошибка: {str(e)}")
    
    def save_images(self):
        df = pandas.read_excel(self.excel_path)
        img = cv2.imread(self.base_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем изображение в формат RGB для Pillow
        output_folder = Path(self.export_directory) / "result_images"
        output_folder.mkdir(parents=True, exist_ok=True)

        for _, row in df.iterrows():
            filename = Path(row["anomaly_filename"]).name
            img_path = output_folder / filename

            anomaly_type = row["Y"]
            x, y, w, h = row["bounding_rect_x"], row["bounding_rect_y"], row["bounding_rect_w"], row["bounding_rect_h"]

            colour = ast.literal_eval(row["anomaly_colour"]) 

            output_image = img.copy()
            cv2.rectangle(output_image, (x, y), (x + w, y + h), colour, 2)

            # Сохраняем через Pillow
            pil_img = Image.fromarray(output_image)  # Преобразуем в изображение Pillow
            pil_img.save(str(img_path))  # Сохраняем через Pillow

    def on_export_clicked(self):
        """Обработчик нажатия на кнопку 'Выгрузить статистику'"""
        # Проверка, что директория не пуста
        if not os.listdir(self.output_folder):  
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Директория пуста. Невозможно выполнить выгрузку.")
            return  
        
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выгрузить статистику", "")
        if directory:
            self.export_directory = Path(directory)
            print(f"Выбрана директория для выгрузки: {self.export_directory}")
            output_dir = os.path.join(self.export_directory, 'result.xlsx')
            try:
                if not os.path.exists(self.excel_path):
                    QtWidgets.QMessageBox.critical(self, "Ошибка", f"Файл {self.excel_path} не найден!")
                    return
                # Загружаем таблицу
                df = pandas.read_excel(self.excel_path)
                # Заменяем в столбце 'anomaly_filename' путь '/output_folder/' на новый путь
                if 'anomaly_filename' in df.columns:
                    output_folder = Path(self.export_directory) / "result_images"
                    df['anomaly_filename'] = df['anomaly_filename'].apply(
                        lambda x: str(output_folder / Path(x).name) if isinstance(x, str) else x
                    )
                # Сохраняем по новому пути
                df.to_excel(output_dir, index=False)
                self.save_images()
                QtWidgets.QMessageBox.information(self, "Успех", "Статистика успешно сохранена.")
            except PermissionError as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Ошибка при сохранении файла.")
        else:
            print("Директория не выбрана")

    def on_set_directory_clicked(self):
        """Обработчик нажатия на кнопку 'Указать директорию'"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Указать директорию", "")
        if directory:
            self.selected_directory = Path(directory)
            print(f"Выбрана директория: {self.selected_directory}")
            self.load_images_to_view()
            # Передаем путь в character.py
            # set_reference_path(self.selected_directory)
        else:
            print("Директория не выбрана")

    def on_run_analysis_clicked(self):
        """Обработчик нажатия на кнопку 'Провести анализ'"""
        selected_row = self.table_model.checked_row
        if selected_row == -1:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите шаблон для анализа.")
            return

        template_directory = self.template_df.iloc[selected_row]["Директория эталонов"]

        if not self.selected_directory or not self.selected_directory.exists():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите корректную директорию для анализа.")
            return

        if not Path(template_directory).exists():
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Директория шаблона не существует: {template_directory}")
            return

        print("🔄 Очистка старых изображений перед анализом...")

        # Очищаем папку output_folder перед анализом
        for file_name in os.listdir(self.output_folder):
            file_path = os.path.join(self.output_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Ошибка при удалении {file_path}: {e}")

        print("✅ Анализ начался...")
        if os.path.exists(self.excel_path):
            try:
                os.remove(self.excel_path)  # Удаляем файл
                print(f"Файл {self.excel_path} успешно удален.")
            except PermissionError:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Файл {self.excel_path} уже открыт. Закройте его и попробуйте снова.")
                return
        else:
            print(f"Файл {self.excel_path} не существует.")
        # Запуск анализа
        set_reference_path(self.selected_directory, template_directory)

        QtWidgets.QMessageBox.information(self, "Успех", "Анализ успешно проведен.")

        # ⬇️ ДОБАВЛЯЕМ ВЫЗОВ ОБНОВЛЕНИЯ СПИСКА ИЗОБРАЖЕНИЙ
        print("🔄 Перезапускаем загрузку изображений после анализа...")
        
        self.image_navigator.load_current_image()
    
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

    def update_image_display(self):
        """Проверка папки и обновление отображения изображений."""
        if not os.path.exists(self.output_folder):
            return

        print("Обновление списка изображений...")
        files = sorted(
            [f for f in os.listdir(self.output_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: os.path.getmtime(os.path.join(self.output_folder, x))
        )

        new_files = [f for f in files if f not in self.processed_files]

        if new_files:
            self.processed_files.update(new_files)
            self.display_all_images()
        else:
            print("Новых изображений не найдено.")

    def display_all_images(self):
        """Отображает одно изображение с bounding box из Excel"""

        if not os.path.exists(self.output_folder):
            print(f"Папка {self.output_folder} не существует.")
            return

        # Загружаем данные из Excel
        df = pandas.read_excel(self.excel_path)

        # Проверяем, есть ли данные
        if df.empty:
            QtWidgets.QMessageBox.information(self, "Информация", "Нет данных для отображения.")
            return

        # Берем путь до одного базового изображения
        if not os.path.exists(self.base_image_path):
            print(f"Файл {self.base_image_path} не найден.")
            return

        # Загружаем изображение
        pixmap = QtGui.QPixmap(self.base_image_path)

        if pixmap.isNull():
            print(f"Ошибка загрузки изображения: {self.base_image_path}")
            return

        # 🚀 ИСПРАВЛЕНИЕ! Удаляем `setScene(scene)`, используем `setPixmap()`
        self.ui.defect.setPixmap(pixmap)
        self.ui.defect.setScaledContents(True)  # Подгоняем изображение по размеру

    def load_first_image(self):
        """Очищает список и отображает только первое изображение"""
        if hasattr(self.ui, "image_list"):  # Проверяем, что image_list существует
            self.ui.image_list.clear()  # Очищаем список

            if self.image_navigator.image_files:
                first_image = self.image_navigator.image_files[0]  # Берем первое изображение
                self.ui.image_list.addItem(first_image)  # Добавляем в список
        else:
            print("Ошибка: image_list не найден в Ui_MainWindow")


    def load_images_to_view(self):
        """Загружает изображения из текущей директории и отображает их в QListWidget"""
        if not self.selected_directory or not self.selected_directory.exists():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Директория не выбрана или не существует.")
            return

        image_files = [file for file in self.selected_directory.iterdir() if IMAGE_PATTERN.match(file.name)]

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
            widget_width = self.ui.image_list.width() - LIST_WIDGET_PADDING  # Учитываем отступы
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
