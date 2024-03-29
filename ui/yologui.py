# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1240, 730)
        font = QtGui.QFont()
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Loadmodelbt = QtWidgets.QPushButton(self.centralwidget)
        self.Loadmodelbt.setGeometry(QtCore.QRect(820, 10, 131, 31))
        self.Loadmodelbt.setObjectName("Loadmodelbt")
        self.resultdisplay = QtWidgets.QLabel(self.centralwidget)
        self.resultdisplay.setGeometry(QtCore.QRect(20, 60, 931, 601))
        self.resultdisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.resultdisplay.setObjectName("resultdisplay")
        self.resulttext = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.resulttext.setGeometry(QtCore.QRect(960, 60, 271, 211))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.resulttext.setFont(font)
        self.resulttext.setObjectName("resulttext")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(960, 290, 251, 80))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.conf_slider = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setProperty("value", 60)
        self.conf_slider.setOrientation(QtCore.Qt.Horizontal)
        self.conf_slider.setObjectName("conf_slider")
        self.horizontalLayout.addWidget(self.conf_slider)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.plotmode = QtWidgets.QComboBox(self.centralwidget)
        self.plotmode.setGeometry(QtCore.QRect(960, 420, 251, 31))
        self.plotmode.setObjectName("plotmode")
        self.plotmode.addItem("")
        self.plotmode.addItem("")
        self.plotmode.addItem("")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(960, 380, 249, 36))
        self.label_3.setObjectName("label_3")
        self.startbt = QtWidgets.QPushButton(self.centralwidget)
        self.startbt.setGeometry(QtCore.QRect(960, 470, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.startbt.setFont(font)
        self.startbt.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.startbt.setObjectName("startbt")
        self.stopbt = QtWidgets.QPushButton(self.centralwidget)
        self.stopbt.setGeometry(QtCore.QRect(1050, 470, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.stopbt.setFont(font)
        self.stopbt.setStyleSheet("background-color: rgb(255, 0, 0);")
        self.stopbt.setObjectName("stopbt")
        self.dir_show = QtWidgets.QLineEdit(self.centralwidget)
        self.dir_show.setGeometry(QtCore.QRect(10, 10, 801, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.dir_show.setFont(font)
        self.dir_show.setText("")
        self.dir_show.setObjectName("dir_show")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1240, 27))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.conf_slider.valueChanged['int'].connect(self.label_2.setNum) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Loadmodelbt.setText(_translate("MainWindow", "Selected model"))
        self.resultdisplay.setText(_translate("MainWindow", "Result Display"))
        self.label.setText(_translate("MainWindow", "Confidence Threshold"))
        self.label_2.setText(_translate("MainWindow", "60"))
        self.plotmode.setItemText(0, _translate("MainWindow", "Class"))
        self.plotmode.setItemText(1, _translate("MainWindow", "Class and Confidence"))
        self.plotmode.setItemText(2, _translate("MainWindow", "Count"))
        self.label_3.setText(_translate("MainWindow", "Plot mode"))
        self.startbt.setText(_translate("MainWindow", "Start"))
        self.stopbt.setText(_translate("MainWindow", "Stop"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
