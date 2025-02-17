from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def button_style(self, color):
        """Стиль для кнопок"""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: 2px solid #4C566A;
                border-radius: 5px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: #4C566A;
                color: {color};
            }}
        """

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(873, 571)

        # Применяем стиль для MainWindow
        MainWindow.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
                color: #ECEFF4;
                font-family: 'Arial';
                font-size: 14px;
            }
        """)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # Виджет "Настройки"
        self.setting = QtWidgets.QWidget(self.centralwidget)
        self.setting.setObjectName("setting")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.setting)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.defect = QtWidgets.QLabel(self.setting)
        self.defect.setObjectName("defect")
        self.defect.setAlignment(QtCore.Qt.AlignCenter)  # Центрируем изображение
        self.defect.setStyleSheet("""
            QLabel {
                border: 2px solid #88C0D0;
                background-color: #3B4252;
            }
        """)

        self.verticalLayout_3.addWidget(self.defect)

        from PyQt5.QtWidgets import QSizePolicy

        # Горизонтальный контейнер для кнопок
        self.button_layout = QtWidgets.QHBoxLayout()

        # Кнопка "Влево"
        self.btn_left = QtWidgets.QPushButton(self.setting)
        self.btn_left.setText("⬅️")
        self.btn_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Растягиваем по ширине
        self.btn_left.setFixedHeight(40)  # Высота фиксированная
        self.btn_left.setStyleSheet("""
            QPushButton {
                background-color: #2E3440;
                color: #88C0D0;
                border: 2px solid #88C0D0;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3B4252;
            }
        """)
        self.button_layout.addWidget(self.btn_left)

        # Кнопка "Вправо"
        self.btn_right = QtWidgets.QPushButton(self.setting)
        self.btn_right.setText("➡️")
        self.btn_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Растягиваем по ширине
        self.btn_right.setFixedHeight(40)  # Высота фиксированная
        self.btn_right.setStyleSheet("""
            QPushButton {
                background-color: #2E3440;
                color: #88C0D0;
                border: 2px solid #88C0D0;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3B4252;
            }
        """)
        self.button_layout.addWidget(self.btn_right)

        # ⬇️ Добавляем кнопки "Влево" и "Вправо" прямо ПЕРЕД кнопкой "Изменить"
        self.verticalLayout_3.addLayout(self.button_layout)

        self.edit = QtWidgets.QPushButton(self.setting)
        self.edit.setObjectName("edit")
        self.edit.setStyleSheet(self.button_style("#88C0D0"))
        self.verticalLayout_3.addWidget(self.edit)

        self.cancel = QtWidgets.QPushButton(self.setting)
        self.cancel.setObjectName("cancel")
        self.cancel.setStyleSheet(self.button_style("#BF616A"))
        self.verticalLayout_3.addWidget(self.cancel)

        self.save = QtWidgets.QPushButton(self.setting)
        self.save.setObjectName("save")
        self.save.setStyleSheet(self.button_style("#A3BE8C"))
        self.verticalLayout_3.addWidget(self.save)

        self.horizontalLayout.addWidget(self.setting)

        # Виджет "Изображения"
        self.imagine = QtWidgets.QWidget(self.centralwidget)
        self.imagine.setObjectName("imagine")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.imagine)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.imagine_label = QtWidgets.QLabel(self.imagine)
        self.imagine_label.setObjectName("imagine_label")
        self.imagine_label.setStyleSheet("""
            QLabel {
                color: #D8DEE9;
                font-weight: bold;
                font-size: 18px;
            }
        """)
        self.verticalLayout_2.addWidget(self.imagine_label)

        # Добавляем image_list
        self.image_list = QtWidgets.QListWidget(self.imagine)
        self.image_list.setObjectName("image_list")
        self.image_list.setViewMode(QtWidgets.QListWidget.IconMode)
        self.image_list.setIconSize(QtCore.QSize(100, 100))
        self.image_list.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.image_list.setSpacing(10)
        self.image_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #81A1C1;
                background-color: #3B4252;
                color: #D8DEE9;
            }
        """)
        self.verticalLayout_2.addWidget(self.image_list)

        self.set_directory = QtWidgets.QPushButton(self.imagine)
        self.set_directory.setObjectName("set_directory")
        self.set_directory.setStyleSheet(self.button_style("#88C0D0"))
        self.verticalLayout_2.addWidget(self.set_directory)

        self.analysis = QtWidgets.QPushButton(self.imagine)
        self.analysis.setEnabled(True)
        self.analysis.setObjectName("analysis")
        self.analysis.setStyleSheet(self.button_style("#81A1C1"))
        self.verticalLayout_2.addWidget(self.analysis)

        self.statistics = QtWidgets.QPushButton(self.imagine)
        self.statistics.setEnabled(True)
        self.statistics.setObjectName("statistics")
        self.statistics.setStyleSheet(self.button_style("#5E81AC"))
        self.verticalLayout_2.addWidget(self.statistics)

        self.horizontalLayout.addWidget(self.imagine)

        # Секция "Шаблоны"
        self.template_3 = QtWidgets.QWidget(self.centralwidget)
        self.template_3.setObjectName("template_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.template_3)
        self.verticalLayout.setObjectName("verticalLayout")

        self.template_label = QtWidgets.QLabel(self.template_3)
        self.template_label.setObjectName("template_label")
        self.template_label.setStyleSheet("""
            QLabel {
                color: #D8DEE9;
                font-weight: bold;
                font-size: 18px;
            }
        """)
        self.verticalLayout.addWidget(self.template_label)

        self.template_2 = QtWidgets.QTableView(self.template_3)
        self.template_2.setObjectName("template_2")
        self.template_2.setStyleSheet("""
            QTableView {
                border: 2px solid #88C0D0;
                background-color: #3B4252;
                color: #D8DEE9;
                gridline-color: #88C0D0;
            }
        """)
        self.verticalLayout.addWidget(self.template_2)

        self.delete_template = QtWidgets.QPushButton(self.template_3)
        self.delete_template.setObjectName("delete_template")
        self.delete_template.setStyleSheet(self.button_style("#BF616A"))
        self.verticalLayout.addWidget(self.delete_template)

        self.insert_template = QtWidgets.QPushButton(self.template_3)
        self.insert_template.setObjectName("insert_template")
        self.insert_template.setStyleSheet(self.button_style("#A3BE8C"))
        self.verticalLayout.addWidget(self.insert_template)

        self.horizontalLayout.addWidget(self.template_3)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Stylish Interface"))
        self.edit.setText(_translate("MainWindow", "Изменить"))
        self.btn_left.setText(_translate("MainWindow", "<-"))
        self.btn_right.setText(_translate("MainWindow", "->"))
        self.cancel.setText(_translate("MainWindow", "Отменить"))
        self.save.setText(_translate("MainWindow", "Сохранить"))
        self.imagine_label.setText(_translate("MainWindow", "Изображения"))
        self.set_directory.setText(_translate("MainWindow", "Указать директорию"))
        self.analysis.setText(_translate("MainWindow", "Провести анализ"))
        self.statistics.setText(_translate("MainWindow", "Выгрузить статистику"))
        self.template_label.setText(_translate("MainWindow", "Шаблоны"))
        self.delete_template.setText(_translate("MainWindow", "Удалить шаблон"))
        self.insert_template.setText(_translate("MainWindow", "Добавить шаблон"))


class Ui_SelectTemplateDialog(object):
    """Интерфейс диалогового окна для добавления шаблона"""
    def setupUi(self, Dialog):
        Dialog.setObjectName("AddTemplateDialog")
        Dialog.resize(500, 400)

        # Применяем стиль для диалогового окна
        Dialog.setStyleSheet("""
            QDialog {
                background-color: #2E3440;
                color: #ECEFF4;
                font-family: 'Arial';
                font-size: 14px;
            }
            QLabel {
                color: #D8DEE9;
            }
            QLineEdit, QTableWidget {
                background-color: #3B4252;
                color: #ECEFF4;
                border: 1px solid #88C0D0;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #88C0D0;
                color: white;
                border: 2px solid #4C566A;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #4C566A;
                color: #88C0D0;
            }
        """)

        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")

        self.table = QtWidgets.QTableWidget(Dialog)
        self.table.setObjectName("table")
        self.table.setColumnCount(2)
        self.table.setRowCount(1)
        self.table.setHorizontalHeaderLabels(["Наименование", "Код изделия"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.verticalLayout.addWidget(self.table)

        self.directoryLayout = QtWidgets.QHBoxLayout()
        self.directoryLabel = QtWidgets.QLabel(Dialog)
        self.directoryLabel.setText("Директория эталонов:")
        self.directoryLayout.addWidget(self.directoryLabel)

        self.directoryInput = QtWidgets.QLineEdit(Dialog)
        self.directoryInput.setObjectName("directoryInput")
        self.directoryLayout.addWidget(self.directoryInput)

        self.browseButton = QtWidgets.QPushButton(Dialog)
        self.browseButton.setText("Обзор")
        self.browseButton.setObjectName("browseButton")
        self.directoryLayout.addWidget(self.browseButton)

        self.verticalLayout.addLayout(self.directoryLayout)

        self.preview_label = QtWidgets.QLabel(Dialog)
        self.preview_label.setObjectName("preview_label")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #3B4252;
                border: 1px solid #88C0D0;
                padding: 10px;
                min-height: 150px;
            }
        """)
        self.preview_label.setText("Предпросмотр")
        self.verticalLayout.addWidget(self.preview_label)

        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton(Dialog)
        self.save_button.setText("Сохранить")
        self.buttonLayout.addWidget(self.save_button)

        self.cancel_button = QtWidgets.QPushButton(Dialog)
        self.cancel_button.setText("Отмена")
        self.buttonLayout.addWidget(self.cancel_button)

        self.verticalLayout.addLayout(self.buttonLayout)
