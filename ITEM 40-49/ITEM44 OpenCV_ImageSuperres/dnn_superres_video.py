# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 21:08:22 2020

@author: luohenyueji
视频超分放大
"""
import cv2

from cv2 import dnn_superres


def main():
    input_path = "./video/chaplin.mp4"
    output_path = "./video/out_chaplin.mp4"
    # 选择模型 edsr, espcn, fsrcnn or lapsrn
    algorithm = "lapsrn"
    # 放大比例，2，3，4，8，根据模型结构选择
    scale = 2
    # 模型路径
    path = "./model/LapSRN_x2.pb"

    # 打开视频
    input_video = cv2.VideoCapture(input_path)
    # 输入图像编码尺寸

    ex = int(input_video.get(cv2.CAP_PROP_FOURCC))

    # 获得输出视频图像尺寸
    # 如果视频没有打开
    if input_video is None:
        print("Could not open the video.")
        return

    S = (
    int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale, int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

    output_video = cv2.VideoWriter(output_path, ex, input_video.get(cv2.CAP_PROP_FPS), S, True)

    # 读取超分放大模型
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path)
    sr.setModel(algorithm, scale)

    while True:
        ret, frame = input_video.read()  # 捕获一帧图像

        if not ret:
            print("read video error")
            return
        # 上采样图像
        output_frame = sr.upsample(frame)
        output_video.write(output_frame)

        cv2.namedWindow("Upsampled video", cv2.WINDOW_AUTOSIZE);
        cv2.imshow("Upsampled video", output_frame)

        cv2.namedWindow("Original video", cv2.WINDOW_AUTOSIZE);
        cv2.imshow("Original video", frame)

        c = cv2.waitKey(1);
        # esc退出
        if 27 == c:
            break

    input_video.release()
    output_video.release()


if __name__ == '__main__':
    main()
