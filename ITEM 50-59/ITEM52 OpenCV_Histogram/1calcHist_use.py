import cv2
import numpy as np


def main():
    imgpath = "image/lena.jpg"
    src = cv2.imread(imgpath)
    if src is None:
        print('Could not open or find the image:', imgpath)
        return -1
    bgr_planes = cv2.split(src)
    histSize = 256
    # 256会被排除
    histRange = (0, 256)
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [
                         histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [
                         histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [
                         histSize], histRange, accumulate=accumulate)
    
	# b_hist表示每个像素范围的像素值个数，其总和等于输入图像长乘宽。
	# 如果要统计每个像素范围的像素值百分比，计算方式如下
    assert(sum(b_hist) == src.shape[0] *src.shape[1])
    # b_hist /= sum(b_hist)
    # g_hist /= sum(g_hist)
    # r_hist /= sum(r_hist)
    # assert(sum(b_hist) == 1)
    
    # 以下是绘图代码
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w/histSize))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h,
                 norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h,
                 norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h,
                 norm_type=cv2.NORM_MINMAX)
    for i in range(1, histSize):
        cv2.line(histImage, (bin_w*(i-1), hist_h - int(np.round(b_hist[i-1]))),
                (bin_w*(i), hist_h - int(np.round(b_hist[i]))),
                (255, 0, 0), thickness=2)
        cv2.line(histImage, (bin_w*(i-1), hist_h - int(np.round(g_hist[i-1]))),
                (bin_w*(i), hist_h - int(np.round(g_hist[i]))),
                (0, 255, 0), thickness=2)
        cv2.line(histImage, (bin_w*(i-1), hist_h - int(np.round(r_hist[i-1]))),
                (bin_w*(i), hist_h - int(np.round(r_hist[i]))),
                (0, 0, 255), thickness=2)
    cv2.imshow('src image', src)
    cv2.imshow('dst image', histImage)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()