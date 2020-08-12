# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:20:56 2020

@author: luohenyueji
"""

import cv2
from time import *

# TODO 背景减除算法集合
ALGORITHMS_TO_EVALUATE = [
    (cv2.bgsegm.createBackgroundSubtractorGMG(20, 0.7), 'GMG'),
    (cv2.bgsegm.createBackgroundSubtractorCNT(), 'CNT'),
    (cv2.createBackgroundSubtractorKNN(), 'KNN'),
    (cv2.bgsegm.createBackgroundSubtractorMOG(), 'MOG'),
    (cv2.createBackgroundSubtractorMOG2(), 'MOG2'),
    (cv2.bgsegm.createBackgroundSubtractorGSOC(), 'GSOC'),
    (cv2.bgsegm.createBackgroundSubtractorLSBP(), 'LSBP'),
]


# TODO 主函数
def main():
    # 背景分割识别器序号
    algo_index = 0
    subtractor = ALGORITHMS_TO_EVALUATE[algo_index][0]
    videoPath = "./video/vtest.avi"
    show_fgmask = False

    # 获得运行环境CPU的核心数
    nthreads = cv2.getNumberOfCPUs()
    # 设置线程数
    cv2.setNumThreads(nthreads)

    # 读取视频
    capture = cv2.VideoCapture(videoPath)

    # 当前帧数
    frame_num = 0
    # 总执行时间
    sum_Time = 0.0

    while True:
        ret, frame = capture.read()
        if not ret:
            return
        begin_time = time()
        fgmask = subtractor.apply(frame)
        end_time = time()
        run_time = end_time - begin_time
        sum_Time = sum_Time + run_time
        # 平均执行时间
        average_Time = sum_Time / (frame_num + 1)

        if show_fgmask:
            segm = fgmask
        else:
            segm = (frame * 0.5).astype('uint8')
            cv2.add(frame, (100, 100, 0, 0), segm, fgmask)

        # 显示当前方法
        cv2.putText(segm, ALGORITHMS_TO_EVALUATE[algo_index][1], (10, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 255),
                    2,
                    cv2.LINE_AA)
        # 显示当前线程数
        cv2.putText(segm, str(nthreads) + " threads", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 255), 2,
                    cv2.LINE_AA)
        # 显示当前每帧执行时间
        cv2.putText(segm, "averageTime {} s".format(average_Time), (10, 90), cv2.FONT_HERSHEY_PLAIN, 2.0,
                    (255, 0, 255), 2, cv2.LINE_AA);

        cv2.imshow('some', segm)
        key = cv2.waitKey(1) & 0xFF
        frame_num = frame_num + 1

        # 按'q'健退出循环
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
