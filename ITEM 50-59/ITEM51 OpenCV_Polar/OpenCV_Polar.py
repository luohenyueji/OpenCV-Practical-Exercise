import numpy as np
import math
import cv2

# ----- 全局参数
# PAI值
PI = math.pi
# 设置输入图像固定尺寸（必要）
HEIGHT, WIDTH = 300, 300
# 输入图像圆的半径，一般是宽高一半
CIRCLE_RADIUS = int(HEIGHT / 2)
# 圆心坐标
CIRCLE_CENTER = [HEIGHT / 2, WIDTH / 2]
# 极坐标转换后图像的高，可自己设置
LINE_HEIGHT = int(CIRCLE_RADIUS / 1.5)
# 极坐标转换后图像的宽，一般是原来圆形的周长
LINE_WIDTH = int(2 * CIRCLE_RADIUS * PI)


# ----- 将圆环变为矩形
def create_line_image(img):
    # 建立展开后的图像
    line_image = np.zeros((LINE_HEIGHT, LINE_WIDTH, 3), dtype=np.uint8)
    # 按照圆的极坐标赋值
    for row in range(line_image.shape[0]):
        for col in range(line_image.shape[1]):
            # 角度，最后的-0.2是用于优化结果，可以自行调整
            theta = PI * 2 / LINE_WIDTH * (col + 1)-0.2
            # 半径，减1防止超界
            rho = CIRCLE_RADIUS - row - 1
            
            # ----- 基础变换
            # x = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.0)
            # y = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.0)

            # ----- 任意起始位置变换
            # # 1 确定极坐标
            # x0 = rho * math.cos(theta) + 0.0
            # y0 = rho * math.sin(theta) + 0.0
            
            # # 2 确定旋转角度
            # angle = math.pi* (-120) / 360 * 2 
        
            # # 3 确定直角坐标
            # x1 = x0*math.cos(angle)-y0*math.sin(angle) + 0
            # y1 = x0*math.sin(angle)+y0*math.cos(angle) + 0
            
            # # 4 切换为OpenCV图像坐标
            # x = int(CIRCLE_CENTER[0] + x1)
            # y = int(CIRCLE_CENTER[1] - y1)
            
            
            # ----- 任意起始位置顺时针变换
            # 1 确定极坐标
            x0 = rho * math.sin(theta) + 0.0
            y0 = rho * math.cos(theta) + 0.0
            
            # 2 确定旋转角度
            angle = math.pi* (-150) / 360 * 2 
        
            # 3 确定直角坐标
            x1 = x0*math.cos(angle)-y0*math.sin(angle) + 0
            y1 = x0*math.sin(angle)+y0*math.cos(angle) + 0
            
            # 4 切换为OpenCV图像坐标
            x = int(CIRCLE_CENTER[0] + x1)
            y = int(CIRCLE_CENTER[1] - y1)
            
            
            # 赋值
            line_image[row, col, :] = img[y, x, :]
    # 如果想改变输出图像方向，旋转就行了
    # line_image = cv2.rotate(line_image, cv2.ROTATE_90_CLOCKWISE)
    return line_image


# ----- 主程序
def main(imgpath):
    # 读取图像
    img = cv2.imread(imgpath)
    if img is None:
        print("please check image path")
        return
    # 图像重置为固定大小
    img = cv2.resize(img, (HEIGHT, WIDTH))
    print(img.shape)

    # 展示原图
    cv2.imshow("src", img)
    output = create_line_image(img)
    # 展示结果
    cv2.imshow("dst", output)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 输入图像路径
    imgpath = "./image/clock.jpg"
    main(imgpath)
