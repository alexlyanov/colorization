#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import sys

from PyQt5 import QtGui, QtCore, QtWidgets

from paint_lib import GrayToColored

class Window(QtWidgets.QWidget):
    
    def __init__(self, parent = None):
        super(Window, self).__init__(parent)

        self.button = QtWidgets.QPushButton('Browse image')
        self.button.clicked.connect(self.show_image)
        
        self.image_frame1 = QtWidgets.QLabel()
        self.image_frame2 = QtWidgets.QLabel()

        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.button, 0, 0, 0, 1)
        self.layout.addWidget(self.image_frame1, 1, 0)
        self.layout.addWidget(self.image_frame2, 1, 1)
        self.setLayout(self.layout)
        
        self.model = GrayToColored()

    @QtCore.pyqtSlot()
    
    def show_image(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')[0]
        
        self.image = cv2.imread('images\img1.jpg', cv2.IMREAD_COLOR)
        
        self.result = self.model.getColoredImg(self.image)
        self.result = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame1.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        self.result = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame2.setPixmap(QtGui.QPixmap.fromImage(self.result))

if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    
    window = Window()
    window.show()
    
    sys.exit(app.exec_())


# In[ ]:




