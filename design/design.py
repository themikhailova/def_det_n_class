# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


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

        self.defect = QtWidgets.QGraphicsView(self.setting)
        self.defect.setObjectName("defect")
        self.defect.setStyleSheet("""
            QGraphicsView {
                border: 2px solid #88C0D0;
                background-color: #3B4252;
            }
        """)
        self.verticalLayout_3.addWidget(self.defect)

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

        self.directory = QtWidgets.QListView(self.imagine)
        self.directory.setObjectName("directory")
        self.directory.setStyleSheet("""
            QListView {
                border: 2px solid #81A1C1;
                background-color: #3B4252;
                color: #D8DEE9;
            }
        """)
        self.verticalLayout_2.addWidget(self.directory)

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

        # Удалила автоподключение, т.к. из-за него у нас выводилось все по 3 раза
        # QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Stylish Interface"))
        self.edit.setText(_translate("MainWindow", "Изменить"))
        self.cancel.setText(_translate("MainWindow", "Отменить"))
        self.save.setText(_translate("MainWindow", "Сохранить"))
        self.imagine_label.setText(_translate("MainWindow", "Изображения"))
        self.set_directory.setText(_translate("MainWindow", "Указать директорию"))
        self.analysis.setText(_translate("MainWindow", "Провести анализ"))
        self.statistics.setText(_translate("MainWindow", "Выгрузить статистику"))
        self.template_label.setText(_translate("MainWindow", "Шаблоны"))
        self.delete_template.setText(_translate("MainWindow", "Удалить шаблон"))
        self.insert_template.setText(_translate("MainWindow", "Добавить шаблон"))