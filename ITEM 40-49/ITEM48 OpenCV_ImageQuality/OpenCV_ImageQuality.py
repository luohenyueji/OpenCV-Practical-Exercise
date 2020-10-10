# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 05:27:28 2020

@author: luohenyueji
"""

import cv2
import numpy as np
import time

# ----- 时间装饰器，打印运行结果和运行时间
def usetime(func):
    def inner(*args, **kwargs):
        time_start = time.time()
        # 装饰的函数在此运行
        result = func(*args, **kwargs)
        time_run = time.time() - time_start
        # 打印结果
        print(func.__name__ + '_result is: {:.3f}'.format(result))
        # 打印运行时间
        print(func.__name__ + '_cost time is: {:.3f} s'.format(time_run))

    return inner


# ----- 均方误差 MSE
@usetime
def MSE(img1, img2):
    # 静态方法，一步到位
    # 质量结果图quality_map就是检测图像和基准图像各个像素点差值结果
    result_static, quality_map = cv2.quality.QualityMSE_compute(img1, img2)
    # 另外一种动态计算的方法，但是MSE的计算可能有问题
    # obj = cv2.quality.QualityMSE_create(img1)
    # result = obj.compute(img2)
    # quality_map = obj.getQualityMap()
    # 计算均值
    score = np.mean([i for i in result_static if (i != 0 and not np.isinf(i))])
    score = 0 if np.isnan(score) else score
    return score


# ----- 峰值信噪比 PSNR
@usetime
def PSNR(img1, img2):
    # 静态方法，一步到位
    # 质量结果图quality_map就是检测图像和基准图像各个像素点差值结果
    # maxPixelValue参数为PSNR计算公式中的MAX，即图片可能的最大像素值，通常为255
    result_static, quality_map = cv2.quality.QualityPSNR_compute(img1, img2, maxPixelValue=255)
    # 另外一种动态计算的方法
    # obj = cv2.quality.QualityPSNR_create(img1, maxPixelValue=255)
    # result = obj.compute(img2)
    # quality_map = obj.getQualityMap()
    # 计算均值
    score = np.mean([i for i in result_static if (i != 0 and not np.isinf(i))])
    score = 0 if np.isnan(score) else score
    return score


# ----- 梯度幅度相似性偏差 GMSD
@usetime
def GMSD(img1, img2):
    # 静态方法，一步到位
    # 质量结果图quality_map就是检测图像和基准图像各个像素点差值结果
    result_static, quality_map = cv2.quality.QualityGMSD_compute(img1, img2)
    # 另外一种动态计算的方法
    # obj = cv2.quality.QualityGMSD_create(img1)
    # result = obj.compute(img2)
    # quality_map = obj.getQualityMap()
    # 计算均值
    score = np.mean([i for i in result_static if (i != 0 and not np.isinf(i))])
    score = 0 if np.isnan(score) else score
    return score


# ----- 结构相似性 SSIM
@usetime
def SSIM(img1, img2):
    # 静态方法，一步到位
    # 质量结果图quality_map就是检测图像和基准图像各个像素点差值结果
    result_static, quality_map = cv2.quality.QualitySSIM_compute(img1, img2)
    # 另外一种动态计算的方法
    # obj = cv2.quality.QualitySSIM_create(img1)
    # result = obj.compute(img2)
    # quality_map = obj.getQualityMap()
    # 计算均值
    score = np.mean([i for i in result_static if (i != 0 and not np.isinf(i))])
    score = 0 if np.isnan(score) else score
    return score


# ----- 盲/无参考图像空间质量评估器 BRISQUE
@usetime
def BRISQUE(img):
    # path to the trained model
    model_path = "./model/brisque_model_live.yml"
    # path to range file
    range_path = "./model/brisque_range_live.yml"
    # 静态计算方法
    result_static = cv2.quality.QualityBRISQUE_compute(img, model_path, range_path)
    # # 另外一种动态计算的方法
    # obj = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    # result = obj.compute(img)
    # 计算均值
    score = np.mean([i for i in result_static if (i != 0 and not np.isinf(i))])
    score = 0 if np.isnan(score) else score
    return score


def main():
    # img1为基准图像，img2为检测图像
    img1 = cv2.imread("image/cut-original-rotated-image.jpg")
    img2 = cv2.imread("image/cut-noise-version.jpg")
    if img1 is None or img2 is None:
        print("img empty")
        return
    # 结果越小，检测图像和基准图像的差距越小
    MSE(img1, img2)
    # 结果越小，检测图像和基准图像的差距越小
    PSNR(img1, img2)
    # 结果为一个0到1之间的数，越大表示检测图像和基准图像的差距越小
    GMSD(img1, img2)
    # 结果为一个0到1之间的数，越大表示检测图像和基准图像的差距越小
    SSIM(img1, img2)
    # 结果为一个0到100之间的数，越小表示检测图像质量越好
    BRISQUE(img2)

if __name__ == '__main__':
    main()
