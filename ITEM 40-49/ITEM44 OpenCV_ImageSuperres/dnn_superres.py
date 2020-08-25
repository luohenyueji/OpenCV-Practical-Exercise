# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 20:08:22 2020

@author: luohenyueji
图像超分放大单输出
"""


import cv2
from cv2 import dnn_superres

def main():
    img_path = "./image/image.png"
    # 可选择算法，bilinear, bicubic, edsr, espcn, fsrcnn or lapsrn
    algorithm = "bilinear"
    # 放大比例，可输入值2，3，4
    scale = 4
    # 模型路径
    path = "./model/LapSRN_x4.pb"

    # 载入图像
    img = cv2.imread(img_path)
    # 如果输入的图像为空
    if img is None:
        print("Couldn't load image: " + str(img_path))
        return

    original_img = img.copy()

    # 创建模型
    sr = dnn_superres.DnnSuperResImpl_create()

    if algorithm == "bilinear":
        img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    elif algorithm == "bicubic":
        img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif algorithm == "edsr" or algorithm == "espcn" or algorithm == "fsrcnn" or algorithm == "lapsrn":
        # 读取模型
        sr.readModel(path)
        #  设定算法和放大比例
        sr.setModel(algorithm, scale)
        # 放大图像
        img_new = sr.upsample(img)
    else:
        print("Algorithm not recognized")

    # 如果失败
    if img_new is None:
        print("Upsampling failed")

    print("Upsampling succeeded. \n")

    # Display
    # 展示图片
    cv2.namedWindow("Initial Image", cv2.WINDOW_AUTOSIZE)
    # 初始化图片
    cv2.imshow("Initial Image", img_new)
    cv2.imwrite("./saved.jpg", img_new)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
