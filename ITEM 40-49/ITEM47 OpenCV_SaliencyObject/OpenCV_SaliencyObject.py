# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:22:58 2020

@author: luohenyueji
"""

import cv2
import random


def main():
    # 显著性检测算法
    # 可选：SPECTRAL_RESIDUAL，FINE_GRAINED，BING，BinWangApr2014
    saliency_algorithm = "BING"
    # 检测视频或者图像
    video_name = "video/vtest.avi"
    # video_name = "video/dog.jpg";
    # 起始帧
    start_frame = 0
    # 模型路径
    training_path = "ObjectnessTrainedModel"

    # 如果算法名和视频名为空，停止检测
    if saliency_algorithm is None or video_name is None:
        print("Please set saliency_algorithm and video_name")
        return

    # open the capture
    cap = cv2.VideoCapture(video_name)

    # 设置视频起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 读图
    _, frame = cap.read()
    if frame is None:
        print("Please set saliency_algorithm and video_name")
        return

    image = frame.copy()

    # 根据输入的方法确定检测类型
    if saliency_algorithm.find("SPECTRAL_RESIDUAL") == 0:

        # 检测结果，白色区域表示显著区域
        saliencyAlgorithm = cv2.saliency.StaticSaliencySpectralResidual_create()

        # 计算显著性
        start = cv2.getTickCount()
        success, saliencyMap = saliencyAlgorithm.computeSaliency(image)
        duration = (cv2.getTickCount() - start) / cv2.getTickFrequency()
        print("computeBinaryMap cost time is: {} ms".format(duration * 1000))

        if success:
            # 二值化图像
            start = cv2.getTickCount()
            _, binaryMap = saliencyAlgorithm.computeBinaryMap(saliencyMap)
            duration = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            print("computeBinaryMap cost time is: {} ms".format(duration * 1000))

            cv2.imshow("Saliency Map", saliencyMap)
            cv2.imshow("Original Image", image)
            cv2.imshow("Binary Map", binaryMap)

            # 转换格式才能保存图片
            saliencyMap = (saliencyMap * 255)
            cv2.imwrite("Results/FINE_GRAINED_saliencyMap.jpg", saliencyMap)
            cv2.imwrite("Results/FINE_GRAINED_binaryMap.jpg", binaryMap)
            cv2.waitKey(0)

    # FINE_GRAINED
    elif saliency_algorithm.find("FINE_GRAINED") == 0:
        saliencyAlgorithm = cv2.saliency.StaticSaliencyFineGrained_create()

        # 计算显著性
        start = cv2.getTickCount()
        success, saliencyMap = saliencyAlgorithm.computeSaliency(image)
        duration = (cv2.getTickCount() - start) / cv2.getTickFrequency()
        print("computeBinaryMap cost time is: {} ms".format(duration * 1000))
        if success:
            # 二值化图像
            start = cv2.getTickCount()
            _, binaryMap = saliencyAlgorithm.computeBinaryMap(saliencyMap)
            duration = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            print("computeBinaryMap cost time is: {} ms".format(duration * 1000))

            cv2.imshow("Saliency Map", saliencyMap)
            cv2.imshow("Original Image", image)
            cv2.imshow("Binary Map", binaryMap)

            # 转换格式才能保存图片
            saliencyMap = (saliencyMap * 255)
            cv2.imwrite("Results/FINE_GRAINED_saliencyMap.jpg", saliencyMap)
            cv2.imwrite("Results/FINE_GRAINED_binaryMap.jpg", binaryMap)
            cv2.waitKey(0)

    # objectness bing
    elif saliency_algorithm.find("BING") == 0:
        # 判断模型是否存在
        if training_path is None:
            print("Path of trained files missing! ")
            return
        else:
            saliencyAlgorithm = cv2.saliency.ObjectnessBING_create()
            # 提取模型文件参数
            saliencyAlgorithm.setTrainingPath(training_path)
            # 将算法检测结果保存在Results文件夹内
            saliencyAlgorithm.setBBResDir("Results")
            # 设置非极大值抑制，值越大检测到的目标越少，检测速度越快
            saliencyAlgorithm.setNSS(50)

        # 计算显著性
        start = cv2.getTickCount()
        # 基于三个颜色空间进行检测，可以只检测一个空间，把training_path下其他空间模型删除即可
        # 如只保留ObjNessB2W8MAXBGR前缀的文件，算法耗时只有原来的一半
        success, saliencyMap = saliencyAlgorithm.computeSaliency(image)
        duration = (cv2.getTickCount() - start) / cv2.getTickFrequency()
        print("computeBinaryMap cost time is: {} ms".format(duration * 1000))
        if success:
            # saliencyMap获取检测到的目标个数
            ndet = saliencyMap.shape[0]
            print("Objectness done ", ndet)

            # The result are sorted by objectness. We only use the first maxd boxes here.
            # 目标按可能性从大到小排列，maxd为显示前5个目标，step设置颜色，jitter设置矩形框微调
            maxd = 5
            step = 255 / maxd
            jitter = 9
            draw = image.copy()

            for i in range(0, min(maxd, ndet)):
                # 获得矩形框坐标点
                bb = saliencyMap[i][0]
                # 设定颜色
                col = ((i * step) % 255), 50, 255 - ((i * step) % 255)
                # 矩形框微调
                off = random.randint(-jitter,
                                     jitter), random.randint(-jitter, jitter)
                # 画矩形
                cv2.rectangle(draw, (bb[0] + off[0], bb[1] + off[1]),
                              (bb[2] + off[0], bb[3] + off[1]), col, 2)
                # mini temperature scale
                # 颜色标注
                cv2.rectangle(draw, (20, 20 + i * 10, 10, 10), col, -1)

            # 保存图片
            cv2.imwrite("Results/BING_draw.jpg", draw)
            cv2.imshow("BING", draw)
            cv2.waitKey(0)

    # 需要传入图像建模
    elif saliency_algorithm.find("BinWangApr2014") == 0:
        saliencyAlgorithm = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        # 设置数据结构大小
        saliencyAlgorithm.setImagesize(image.shape[1], image.shape[0])
        # 初始化
        saliencyAlgorithm.init()
        paused = False

        while True:
            if not paused:
                _, frame = cap.read()
                if frame is None:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 计算显著性
                start = cv2.getTickCount()
                success, saliencyMap = saliencyAlgorithm.computeSaliency(frame)
                duration = (cv2.getTickCount() - start) / \
                           cv2.getTickFrequency()
                print("computeBinaryMap cost time is: {} ms".format(duration * 1000))
                cv2.imshow("image", frame)
                # 显示
                cv2.imshow("saliencyMap", saliencyMap * 255)

            c = cv2.waitKey(2)
            c = chr(c) if c != -1 else 0
            if c == 'q':
                break
            if c == 'p':
                paused = not paused

        cv2.destroyAllWindows()
        return


if __name__ == '__main__':
    main()
