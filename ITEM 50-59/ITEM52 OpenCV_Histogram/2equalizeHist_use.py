import cv2

def main():
    imgpath = "image/lena.jpg"
    src = cv2.imread(imgpath)
    if src is None:
        print('Could not open or find the image:', imgpath)
        return -1
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(src)
    cv2.imshow("src image", src)
    cv2.imshow("dst image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()