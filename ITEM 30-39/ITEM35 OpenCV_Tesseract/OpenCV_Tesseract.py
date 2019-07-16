import cv2
import pytesseract


# 图像路径
imPath = 'image/computer-vision.jpg'


# 命令
config = ('-l eng --oem 1 --psm 3')

# Read image from disk 获得彩色图像
im = cv2.imread(imPath, cv2.IMREAD_COLOR)

# Run tesseract OCR on image
text = pytesseract.image_to_string(im, config=config)

# Print recognized text
print(text)
