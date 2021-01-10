#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH_TO_PROTOTXT = "./model/colorization_deploy_v2.prototxt"
PATH_TO_MODEL = "./model/colorization_release_v2.caffemodel"
PATH_TO_POINTS = "./model/pts_in_hull.npy"

class GrayToColored:
    
    #конструктор, в котором загружается модель
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(PATH_TO_PROTOTXT, PATH_TO_MODEL)
        pts = np.load(PATH_TO_POINTS)
        
        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(class8).blobs = [pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    # проверка принадлежности пикселя изображению
    def checkBounds(self, x, y, xx, yy):
        return (x >= 0 and y >= 0 and x < xx and y < yy)
    
    # устранение шумов типа соль-перец на изображении с помощью медианного фильтра
    def medianFilter(self, image, siz):
        diff = []
        for i in range(-siz // 2, siz // 2 + 1, 1):
            diff.append(i)
        
        n = image.shape[0]
        m = image.shape[1]
        
        med_filt_res = image.copy()
        
        for i in range(n):
            for j in range(m):
                pixs = []
                for ii in range(siz):
                    for jj in range(siz):
                        x = i + diff[ii]
                        y = j + diff[jj]
                        if (self.checkBounds(x, y, n, m)):
                            pixs.append(image[x][y])
                
                if (len(pixs) % 2 == 0):
                    med_filt_res[i][j] = int(int(pixs[len(pixs) // 2]) + int(pixs[len(pixs) // 2 - 1])) / 2.0
                else:
                    med_filt_res[i][j] = pixs[len(pixs) // 2]
        
        return med_filt_res.copy()
    
    # устранение шумов типа соль-перец на изображении с помощью консервативного фильтра
    def conservativeFilter(self, image, siz):
        diff = []
        for i in range(-siz // 2, siz // 2 + 1, 1):
            diff.append(i)
        
        n = image.shape[0]
        m = image.shape[1]
        
        cons_filt_res = image.copy()
        
        for i in range(n):
            for j in range(m):
                pixs = []
                for ii in range(siz):
                    for jj in range(siz):
                        x = i + diff[ii]
                        y = j + diff[jj]
                        if (self.checkBounds(x, y, n, m)):
                            pixs.append(image[x][y])
                                
                pixs.remove(image[i, j])
                
                if (len(pixs) == 0):
                    continue
                
                max_value = max(pixs)
                min_value = min(pixs)
                
                if (image[i, j] > max_value):
                    cons_filt_res[i, j] = max_value
                elif (image[i, j] < min_value):
                    cons_filt_res[i, j] = min_value

        return cons_filt_res.copy()
    
    # перевод изображения из RGB в LAB
    def RGBToLAB(self, image):
        scaled = image.astype("float32") / 255.0
        LAB = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        return LAB
        
    # извлечение параметра L с предшествующим машстабированием изображения
    def getL(self, image):
        resized = cv2.resize(image, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        return L
    
    # прогнозирование параметров A и B с учетом параметра L и возврат изображения к исходным размерам
    def getAB(self, L, w, h):
        self.net.setInput(cv2.dnn.blobFromImage(L))
        AB = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        AB = cv2.resize(AB, (h, w))
        return AB
    
    # склеивание параметров L, A и B для получения полноценного LAB-изображения и перевод его в RGB-кодировку
    def createRGBImg(self, LAB, AB):
        L = cv2.split(LAB)[0]
        newLAB = np.concatenate((L[:, :, np.newaxis], AB), axis=2)
        
        RGB = cv2.cvtColor(newLAB, cv2.COLOR_LAB2RGB)
        RGB = np.clip(RGB, 0, 1)
        RGB = (255 * RGB).astype("uint8")
        return RGB
    
    # перекраска черно-белого изображения в RGB
    def getColoredImg(self, image):
        resized = cv2.resize(image, (224, 224))
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        
        gray = self.medianFilter(gray, 3)
        gray = self.conservativeFilter(gray, 5)
        
        gray = cv2.resize(gray, (image.shape[1], image.shape[0]))
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        LAB = self.RGBToLAB(image)
        L = self.getL(LAB)
        AB = self.getAB(L, image.shape[0], image.shape[1])
        RGB = self.createRGBImg(LAB, AB)
        return RGB

