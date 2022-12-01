import cv2


def main():
    # 感兴趣区域图片
    roi = cv2.imread('image/test3.jpg')
    # 目标图片
    target = cv2.imread('image/test2.jpg')
    if roi is None or target is None:
        print('Could not open or find the images!')
        return -1
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # 计算颜色直方图
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 归一化图片
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    # 返回匹配结果图像，dst为一张二值图，白色区域表示匹配到的目标
    dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
    # 应用线性滤波器，理解成去噪就行了
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cv2.filter2D(dst, -1, disc, dst)
    # 阈值过滤
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    # 将thresh转换为3通道图
    thresh = cv2.merge((thresh, thresh, thresh))
    # 从图片中提取感兴趣区域
    res = cv2.bitwise_and(target, thresh)
    cv2.imshow("target", target)
    cv2.imshow("thresh", thresh)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()