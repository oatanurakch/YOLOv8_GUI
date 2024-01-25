from ultralytics import YOLO
import cv2 
import numpy as np
import os
import sys
from pathlib import Path
from time import sleep

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from PyQt5.QtCore import QTimer, Qt, QCoreApplication, QObject, QThread, pyqtSignal

# Get the root path of the project
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # Path to the root folder of the project
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = str(ROOT)

from ui.yologui import Ui_MainWindow
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

from plotting import colors, Annotator

# Capture Hardware
capHardware = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# State running
staterunning = False

# QThread for update frame from camera
class UpdateFrameCV2(QObject):
    frameread = pyqtSignal(np.ndarray)

    def run(self):
        while capHardware.isOpened() and staterunning:
            ret, frame = capHardware.read()
            if ret:
                self.frameread.emit(frame)
                sleep(0)
            else:
                break

# main class for GUI
class mainui(Ui_MainWindow):
    def __init__(self):
        super().setupUi(MainWindow)
        # Selected model
        self.modelDirectory = None
        # Initial model
        self.model = None
        # Initial button interaction
        self.initialSignal()
        # initial value of config
        self.conf = int(self.conf_slider.value()) / 100
        # Plot mode
        self.plotselect = int(self.plotmode.currentIndex())

    # Initial button interaction
    def initialSignal(self):
        # Load model button
        self.Loadmodelbt.clicked.connect(self.getDirmodel)
        # Start button
        self.startbt.clicked.connect(self.PrepairPredictProcess)
        # Stop button
        self.stopbt.clicked.connect(self.StopDetect)
        # Confidence slider
        self.conf_slider.sliderReleased.connect(self.updateConfidence)
        # Update plot mode
        self.plotmode.currentIndexChanged.connect(self.updatePlotMode)

    # update confidence
    def updateConfidence(self): 
        self.conf = int(self.conf_slider.value()) / 100

    # Update plot mode
    def updatePlotMode(self):
        self.plotselect = int(self.plotmode.currentIndex())

    # Get the directory of the model
    def getDirmodel(self):
        global staterunning
        dialog = QFileDialog()
        pathmodel = dialog.getOpenFileName(MainWindow, 'Selected weight file', ROOT, 'Weight file (*.pt)')[0]
        if pathmodel != '':
            # Store the directory of the model
            self.modelDirectory = pathmodel
            # Set Model
            self.model = YOLO(self.modelDirectory)
            # Change state of predict
            staterunning = True
        else:
            self.modelDirectory = None
            # Set model to None
            self.model = None
        # Set the text of the label
        self.dir_show.setText(self.modelDirectory)

    # Start detection
    def PrepairPredictProcess(self):
        if staterunning:
            # Set QThread
            self.threadUpdateFrame = QThread()
            self.workerThreadUpdate = UpdateFrameCV2()
            self.workerThreadUpdate.moveToThread(self.threadUpdateFrame)
            self.threadUpdateFrame.started.connect(self.workerThreadUpdate.run)
            self.workerThreadUpdate.frameread.connect(self.UpdatePredict)
            # Start detect
            self.threadUpdateFrame.start()
        else:
            print('Please select model first')

    # Update predict
    def UpdatePredict(self, src_img):
        if staterunning:
            # Convert color
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            # Predict process
            results = self.Predict(source_img = src_img)
            # get shape
            h, w, ch = results.shape
            # Byte per line
            bytesperlines = ch * w
            # Set Pixmap
            qImg = QImage(results.data, w, h, bytesperlines, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            # Set Pixamp to label
            self.resultdisplay.setPixmap(pixmap)
        else:
            self.resultdisplay.setText('Result Display')

    # Predict Process
    def Predict(self, source_img):
        # Variable for store result of detection
        classnames = []
        confidences = []
        xyxys = []
        colors_box = []
        quantity_color_of_each_object = {}
        list_result_name_xyxy_qty = []
        str_result = ''
        # Predict
        results = self.model.predict(source = source_img,
                                     conf = self.conf,
                                     device = 0,
                                     save = False,
                                     imgsz = 640)
        
        # Get class in model
        names = self.model.names
        # Custom bounding box
        for r in results:
            for i in range(len(r.boxes.cls)):
                # Classname of object
                classnames.append(names[int(r.boxes.cls[i])])
                # Confidence of object
                confidences.append((r.boxes.conf[i]).cpu().numpy())
                # xyxy of object
                xyxys.append((r.boxes.xyxy[i]).cpu().numpy())
                # Color of object
                colors_box.append((colors(int(r.boxes.cls[i]), False)))
                # Count object
                if names[int(r.boxes.cls[i])] not in quantity_color_of_each_object:
                    # Append color and quantity of each object to dictionary
                    color = colors(int(r.boxes.cls[i]), False)
                    quantity_color_of_each_object[names[int(r.boxes.cls[i])]] = {'color' : color, 'qty' : 1}
                    # Append xyxys of each object to list
                    list_result_name_xyxy_qty.append({'name' : names[int(r.boxes.cls[i])], 'xyxys' : [(r.boxes.xyxy[i]).cpu().numpy()], 'qty' : 1})
                else:
                    # Append color and quantity of each object to dictionary
                    quantity_color_of_each_object[names[int(r.boxes.cls[i])]]['qty'] += 1
                    # Append xyxys of each object to list
                    list_result_name_xyxy_qty.append({'name' : names[int(r.boxes.cls[i])], 'xyxys' : [(r.boxes.xyxy[i]).cpu().numpy()], 'qty' : quantity_color_of_each_object[names[int(r.boxes.cls[i])]]['qty']})
        # Append string result
        for key, value in quantity_color_of_each_object.items():
            str_result += '{}: {}\n' .format(key, value.get('qty'))
        # Clear result text
        self.resulttext.clear()
        # Show result text
        self.resulttext.appendPlainText(str_result)
        # Annotate
        annotated_res = source_img.copy()
        # class 0 is class only
        if self.plotselect == 0:
            annotated_res = Annotator(img = annotated_res, 
                                      xyxys = xyxys, 
                                      classnames = classnames, 
                                      confidences = confidences,
                                      colors = colors_box,
                                      ).drawClass()
        # class 1 is class and confidence
        elif self.plotselect == 1:
            annotated_res = Annotator(img = annotated_res, 
                                      xyxys = xyxys, 
                                      classnames = classnames, 
                                      confidences = confidences,
                                      colors = colors_box,
                                      ).drawClassAndConfidence()
        # Class 2 quantity of each object
        elif self.plotselect == 2:
            annotated_res = Annotator(img = annotated_res, 
                                      xyxys = xyxys, 
                                      classnames = classnames, 
                                      confidences = confidences,
                                      colors = colors_box,
                                      ).drawQuantity(classnames_xyxys_count_object = list_result_name_xyxy_qty)
        
        # Return annotated result
        return annotated_res
    
    # Stop detection
    def StopDetect(self):
        global staterunning
        # if state is running
        if staterunning:
            # Stop thread
            self.threadUpdateFrame.quit()
            # Set state to False
            staterunning = False
            # Clear model
            self.model = None
            # Clear directory
            self.modelDirectory = None
            # Set text of label
            self.dir_show.setText(self.modelDirectory)
        else:
            print('Please start detection first')

if __name__ == '__main__':
    obj = mainui()
    MainWindow.show()
    sys.exit(app.exec_())