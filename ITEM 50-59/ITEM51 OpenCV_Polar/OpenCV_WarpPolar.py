import cv2


# ----- 主函数
def main():
    # INTER_LINEAR 双线性插值，WARP_FILL_OUTLIERS填充所有目标图像像素
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    # 读图
    imagepath = "image/clock.jpg"
    src = cv2.imread(imagepath)
    if src is None:
        print("Could not initialize capturing...\n")
        return -1

    # 圆心坐标
    center = (float(src.shape[0] / 2), float(src.shape[1] / 2))
    # 圆的半径
    maxRadius = min(center[0], center[1])

    # direct transform
    # linear Polar 极坐标变换, None表示OpenCV根据输入自行决定输出图像尺寸
    lin_polar_img = cv2.warpPolar(src, None, center, maxRadius, flags)
    # semilog Polar 半对数极坐标变换, None表示OpenCV根据输入自行决定输出图像尺寸
    log_polar_img = cv2.warpPolar(src, None, center, maxRadius, flags | cv2.WARP_POLAR_LOG)
    # inverse transform 逆变换
    recovered_lin_polar_img = cv2.warpPolar(lin_polar_img, (src.shape[0], src.shape[1]), center, maxRadius,
                                            flags | cv2.WARP_INVERSE_MAP)
    recovered_log_polar = cv2.warpPolar(log_polar_img, (src.shape[0], src.shape[1]), center, maxRadius,
                                        flags | cv2.WARP_POLAR_LOG | cv2.WARP_INVERSE_MAP)

    # 改变结果方向
    # lin_polar_img = cv2.rotate(lin_polar_img, cv2.ROTATE_90_CLOCKWISE)

    # 展示图片
    cv2.imshow("Src frame", src)
    cv2.imshow("Log-Polar", log_polar_img)
    cv2.imshow("Linear-Polar", lin_polar_img)
    cv2.imshow("Recovered Linear-Polar", recovered_lin_polar_img)
    cv2.imshow("Recovered Log-Polar", recovered_log_polar)
    cv2.waitKey(0)
    return 0


if __name__ == '__main__':
    main()
