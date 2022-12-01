import cv2

def main():
    imgs = ["image/lena.jpg", "image/lena_resize.jpg", "image/lena_flip.jpg","image/test.jpg"]
    src_base = cv2.imread(imgs[0])
    src_test1 = cv2.imread(imgs[1])
    src_test2 = cv2.imread(imgs[2])
    src_test3 = cv2.imread(imgs[3])
    if src_base is None or src_test1 is None or src_test2 is None or src_test3 is None:
        print('Could not open or find the images!')
        exit(0)
    # 将图片转换到hsv空间
    hsv_base = cv2.cvtColor(src_base, cv2.COLOR_BGR2HSV)
    hsv_test1 = cv2.cvtColor(src_test1, cv2.COLOR_BGR2HSV)
    hsv_test2 = cv2.cvtColor(src_test2, cv2.COLOR_BGR2HSV)
    hsv_test3 = cv2.cvtColor(src_test3, cv2.COLOR_BGR2HSV)
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue值变化范围为0到179，saturation值变化范围为0到255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    # 合并
    ranges = h_ranges + s_ranges  
    # 使用前两个通道计算直方图
    channels = [0, 1]
    hist_base = cv2.calcHist([hsv_base], channels, None,
                            histSize, ranges, accumulate=False)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_test1 = cv2.calcHist([hsv_test1], channels, None,
                             histSize, ranges, accumulate=False)
    cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_test2 = cv2.calcHist([hsv_test2], channels, None,
                             histSize, ranges, accumulate=False)
    cv2.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_test3 = cv2.calcHist([hsv_test3], channels, None,
                             histSize, ranges, accumulate=False)
    cv2.normalize(hist_test3, hist_test3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    for compare_method in range(6):
        base_base = cv2.compareHist(hist_base, hist_base, compare_method)
        base_test1 = cv2.compareHist(hist_base, hist_test1, compare_method)
        base_test2 = cv2.compareHist(hist_base, hist_test2, compare_method)
        base_test3 = cv2.compareHist(hist_base, hist_test3, compare_method)
        print("method[%s]: base_base : %.3f \t base_test1: %.3f \t base_test2: %.3f \t base_test3: %.3f \n" % (
            compare_method, base_base, base_test1, base_test2, base_test3))
    
    print("Done \n")
    
if __name__ == "__main__":
    main()