# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:48:56 2020

@author: luohenyueji
"""

import cv2
import numpy as np

# ----- 全局变量
# 输入图片
g_image = np.zeros((3, 3, 3), np.uint8)

# gamma变换变量
g_gamma = 40
g_gammaMax = 500
g_gammaWinName = "Gamma Correction"

# 对比度拉伸
g_r1 = 70
g_s1 = 15
g_r2 = 120
g_s2 = 240
g_contrastWinName = "Contrast Stretching"


# 创建gamma变换滑动条
def onTrackbarGamma(x):
    g_gamma = x
    gamma = g_gamma / 100.0
    g_imgGamma = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.gammaCorrection(g_image, g_imgGamma, gamma)
    cv2.imshow(g_gammaWinName, g_imgGamma);
    print(g_gammaWinName + ": " + str(rmsContrast(g_imgGamma)))


# 创建对数变换滑动条
def onTrackbarContrastR1(x):
    g_r1 = x
    g_contrastStretch = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2)
    cv2.imshow("Contrast Stretching", g_contrastStretch)
    print(g_contrastWinName + ": " + str(rmsContrast(g_contrastStretch)))


def onTrackbarContrastS1(x):
    g_s1 = x
    g_contrastStretch = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2)
    cv2.imshow("Contrast Stretching", g_contrastStretch)
    print(g_contrastWinName + ": " + str(rmsContrast(g_contrastStretch)))


def onTrackbarContrastR2(x):
    g_r2 = x
    g_contrastStretch = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2)
    cv2.imshow("Contrast Stretching", g_contrastStretch)
    print(g_contrastWinName + ": " + str(rmsContrast(g_contrastStretch)))


def onTrackbarContrastS2(x):
    g_s2 = x
    g_contrastStretch = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2)
    cv2.imshow("Contrast Stretching", g_contrastStretch)
    print(g_contrastWinName + ": " + str(rmsContrast(g_contrastStretch)))


# 计算对比度
def rmsContrast(scrImg):
    dstImg = cv2.cvtColor(scrImg, cv2.COLOR_BGR2GRAY)
    contrast = dstImg.std()
    return contrast


def main():
    # 图像路径
    inputFilename = "./image/car.png"
    # 读图
    global g_image
    g_image = cv2.imread(inputFilename)
    if g_image is None:
        print("image is empty")
        return

    # 创建滑动条
    cv2.namedWindow(g_gammaWinName)
    # 创建gamma变换筛选方法
    cv2.createTrackbar("Gamma value", g_gammaWinName, g_gamma, g_gammaMax, onTrackbarGamma)

    # 对比度拉伸 Contrast Stretching
    cv2.namedWindow(g_contrastWinName)
    cv2.createTrackbar("Contrast R1", g_contrastWinName, g_r1, 256, onTrackbarContrastR1)
    cv2.createTrackbar("Contrast S1", g_contrastWinName, g_s1, 256, onTrackbarContrastS1)
    cv2.createTrackbar("Contrast R2", g_contrastWinName, g_r2, 256, onTrackbarContrastR2)
    cv2.createTrackbar("Contrast S2", g_contrastWinName, g_s2, 256, onTrackbarContrastS2)

    # Apply intensity transformations
    # 应用强度转换
    # autoscaling
    imgAutoscaled = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.autoscaling(g_image, imgAutoscaled)
    # gamma变换
    g_imgGamma = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.gammaCorrection(g_image, g_imgGamma, g_gamma / 100.0)
    # 对数变换
    imgLog = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.logTransform(g_image, imgLog)
    # 对比度拉伸
    g_contrastStretch = np.zeros(g_image.shape, np.uint8)
    cv2.intensity_transform.contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2)

    # 展示结果
    cv2.imshow("Original Image", g_image);
    print("Original Image: " + str(rmsContrast(g_image)))
    cv2.imshow("Autoscale", imgAutoscaled)
    print("Autoscale: " + str(rmsContrast(imgAutoscaled)))

    cv2.imshow(g_gammaWinName, g_imgGamma)
    print(g_gammaWinName + ": " + str(rmsContrast(g_imgGamma)))

    cv2.imshow("Log Transformation", imgLog)
    print("Log Transformation: " + str(rmsContrast(imgLog)))

    cv2.imshow(g_contrastWinName, g_contrastStretch)
    print(g_contrastWinName + ": " + str(rmsContrast(g_contrastStretch)))

    cv2.waitKey(0)


if __name__ == '__main__':
    main()



