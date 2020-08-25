# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 22:08:22 2020

@author: luohenyueji
不同图像超分算法效果评估
"""

import cv2
from cv2 import dnn_superres
import numpy as np


# TODO 绘图
def showBenchmark(imgs, titles, psnrValues, ssimValues):
    # 绘图
    for i in range(0, len(imgs)):
        # 标题绘图
        cv2.putText(imgs[i], titles[i], (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 0, 255), 2, cv2.LINE_AA)
        # psnr值
        cv2.putText(imgs[i], "PSNR: " + str(psnrValues[i]), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 0, 255), 2, cv2.LINE_AA)
        # ssim值
        cv2.putText(imgs[i], "SSIM: " + str(ssimValues[i]), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 0, 255), 2, cv2.LINE_AA)

    # 图片拼接展示
    img = np.vstack([np.hstack([imgs[0], imgs[1]]), np.hstack([imgs[2], imgs[3]])])
    cv2.imshow("Quality benchmark", img)
    cv2.waitKey(0)


# TODO 图像质量评估
def getQualityValues(upsampled, orig):
    psnr = cv2.PSNR(upsampled, orig)
    q, _ = cv2.quality.QualitySSIM_compute(upsampled, orig)
    ssim = (q[0] + q[1] + q[2]) / 3
    return round(psnr, 3), round(ssim, 3)


def main():
    # 图片路径
    img_path = "./image/butterfly.png"
    # 算法名称 edsr, espcn, fsrcnn or lapsrn
    algorithm = "lapsrn"
    # 模型路径，根据算法确定
    model = "./model/LapSRN_x2.pb"
    # 放大系数
    scale = 2
    psnrValues = []
    ssimValues = []

    img = cv2.imread(img_path)

    if img is None:
        print("Couldn't load image: " + str(img_path))

    # Crop the image so the images will be aligned
    # 裁剪图像，使图像对齐
    width = img.shape[0] - (img.shape[0] % scale)
    height = img.shape[1] - (img.shape[1] % scale)
    cropped = img[0:width, 0:height]

    # Downscale the image for benchmarking
    # 缩小图像，以实现基准质量测试
    img_downscaled = cv2.resize(cropped, None, fx=1.0 / scale, fy=1.0 / scale)

    # Make dnn super resolution instance
    # 超分模型初始化
    sr = dnn_superres.DnnSuperResImpl_create()

    # Read and set the dnn model
    # 读取和设定模型
    sr.readModel(model)
    sr.setModel(algorithm, scale)
    # 放大图像
    img_new = sr.upsample(img_downscaled)

    # DL MODEL
    # 获得模型质量评估值
    psnr, ssim = getQualityValues(cropped, img_new)

    psnrValues.append(psnr)
    ssimValues.append(ssim)

    print(sr.getAlgorithm() + "\n")
    print("PSNR: " + str(psnr) + " SSIM: " + str(ssim) + "\n")
    print("-" * 50)

    # INTER_CUBIC - 三次样条插值放大图像
    bicubic = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    psnr, ssim = getQualityValues(cropped, bicubic)

    psnrValues.append(psnr)
    ssimValues.append(ssim)

    print("Bicubic \n")
    print("PSNR: " + str(psnr) + " SSIM: " + str(ssim) + "\n")
    print("-" * 50)

    # INTER_NEAREST - 最近邻插值

    nearest = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    psnr, ssim = getQualityValues(cropped, nearest)
    psnrValues.append(psnr)
    ssimValues.append(ssim)

    print("Nearest neighbor \n")
    print("PSNR: " + str(psnr) + " SSIM: " + str(ssim) + "\n")
    print("-" * 50)

    # Lanczos插值放大图像

    lanczos = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4);
    psnr, ssim = getQualityValues(cropped, lanczos)
    psnrValues.append(psnr)
    ssimValues.append(ssim)

    print("Lanczos \n")
    print("PSNR: " + str(psnr) + " SSIM: " + str(ssim) + "\n")
    print("-" * 50)

    imgs = [img_new, bicubic, nearest, lanczos]
    titles = [sr.getAlgorithm(), "Bicubic", "Nearest neighbor", "Lanczos"]
    showBenchmark(imgs, titles, psnrValues, ssimValues)


if __name__ == '__main__':
    main()
