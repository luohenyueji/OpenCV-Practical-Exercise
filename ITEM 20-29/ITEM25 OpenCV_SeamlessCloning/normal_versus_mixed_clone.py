
import cv2
import numpy as np

# Read images : src image will be cloned into dst
im = cv2.imread("image/wood-texture.jpg")
obj= cv2.imread("image/iloveyouticket.jpg")

# Create an all white mask
mask = 255 * np.ones(obj.shape, obj.dtype)

# The location of the center of the src in the dst
width, height, channels = im.shape
center = (int(height/2), int(width/2))

# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
monochrome_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MONOCHROME_TRANSFER)
# Write results
cv2.imwrite("opencv-normal-clone-example.jpg", normal_clone)
cv2.imwrite("opencv-mixed-clone-example.jpg", mixed_clone)
cv2.imwrite("opencv-monochrome-clone-example.jpg", monochrome_clone)