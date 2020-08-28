# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:03:21 2020

@author: luohenyueji
"""

import cv2


def test_one(title, a, b):
    # 创建类
    if "AverageHash" == title:
        hashFun = cv2.img_hash.AverageHash_create()
    elif "PHash" == title:
        hashFun = cv2.img_hash.PHash_create()
    elif "MarrHildrethHash" == title:
        hashFun = cv2.img_hash.MarrHildrethHash_create()
    elif "RadialVarianceHash" == title:
        hashFun = cv2.img_hash.RadialVarianceHash_create()
    elif "BlockMeanHash" == title:
        hashFun = cv2.img_hash.BlockMeanHash_create()
    elif "ColorMomentHash" == title:
        hashFun = cv2.img_hash.ColorMomentHash_create()

    tick = cv2.TickMeter()
    print("=== " + title + " ===")

    tick.reset()
    tick.start()
    # # 计算图a的哈希值
    hashA = hashFun.compute(a)
    tick.stop()
    print("compute1: " + str(tick.getTimeMilli()) + " ms")

    tick.reset()
    tick.start()
    # 计算图b的哈希值
    hashB = hashFun.compute(b)
    tick.stop()
    print("compute2: " + str(tick.getTimeMilli()) + " ms")
    # 比较两张图像哈希值的距离
    print("compare: " + str(hashFun.compare(hashA, hashB)))


def main():
    inputImg = cv2.imread("./image/img1.jpg")
    targetImg = cv2.imread("./image/img4.jpg")

    if inputImg is None or targetImg is None:
        print("check input image")
        return

    test_one("AverageHash", inputImg, targetImg)
    test_one("PHash", inputImg, targetImg)
    test_one("MarrHildrethHash", inputImg, targetImg)
    test_one("RadialVarianceHash", inputImg, targetImg)
    test_one("BlockMeanHash", inputImg, targetImg)
    test_one("ColorMomentHash", inputImg, targetImg)


if __name__ == '__main__':
    main()
