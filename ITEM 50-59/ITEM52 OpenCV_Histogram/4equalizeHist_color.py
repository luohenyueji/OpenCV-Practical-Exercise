import cv2

# 颜色通道分别进行均衡化
def equalizeHistChannel(inputImage):
    channels = cv2.split(inputImage)

    # 各个通道图像进行直方图均衡
    cv2.equalizeHist(channels[0], channels[0])
    cv2.equalizeHist(channels[1], channels[1])
    cv2.equalizeHist(channels[2], channels[2])

    # 合并结果
    result = cv2.merge(channels)

    return result

# 仅对亮度通道进行均衡化
def equalizeHistIntensity(inputImage):
    # 将bgr格式转换为yuv444
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YUV)

    channels = cv2.split(inputImage)
    # 均衡化亮度通道
    cv2.equalizeHist(channels[0], channels[0])
    # 合并结果
    result = cv2.merge(channels)
    result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)

    return result


def main():
    imgpath = "image/lena.jpg"
    src = cv2.imread(imgpath)
    if src is None:
        print('Could not open or find the image:', imgpath)
        return -1
    dstChannel = equalizeHistChannel(src)
    dstIntensity = equalizeHistIntensity(src)
    cv2.imshow("src image", src)
    cv2.imshow("dstChannel image", dstChannel)
    cv2.imshow("dstIntensity image", dstIntensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()