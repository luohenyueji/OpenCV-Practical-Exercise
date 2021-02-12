from __future__ import print_function

import numpy as np
import cv2


# ----- 读图
imgL = cv2.imread("images/im0.png",1)
imgL = cv2.resize(imgL,(600,600))

imgR = cv2.imread("images/im1.png",1)
imgR = cv2.resize(imgR,(600,600))

# Setting parameters for StereoSGBM algorithm
# 设置 StereoSGBM相关参数
minDisparity = 0
numDisparities = 64
blockSize = 8
disp12MaxDiff = 1
uniquenessRatio = 10
speckleWindowSize = 10
speckleRange = 8

# Creating an object of StereoSGBM algorithm
# 创建StereoSGBM对象
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
    numDisparities = numDisparities,
    blockSize = blockSize,
    disp12MaxDiff = disp12MaxDiff,
    uniquenessRatio = uniquenessRatio,
    speckleWindowSize = speckleWindowSize,
    speckleRange = speckleRange
)

# Calculating disparith using the StereoSGBM algorithm
# 计算视差
disp = stereo.compute(imgL, imgR).astype(np.float32)

# 结果归一化
disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

# Displaying the disparity map
# 显示结果
cv2.imshow("disparity",disp)
cv2.imshow("left image",imgL)
cv2.imshow("right image",imgR)
cv2.waitKey(0)

cv2.destroyAllWindows()
