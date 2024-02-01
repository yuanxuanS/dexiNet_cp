import cv2
import numpy as np
path = "/home/panpan/DexiNed/blur_resize.jpg"
img = cv2.imread(path, cv2.IMREAD_COLOR)


# Apply 3x3 and 7x7 Gaussian blur
low_sigma = cv2.GaussianBlur(img,(3,3),0)
high_sigma = cv2.GaussianBlur(img,(5,5),0)
# Calculate the DoG by subtracting
dog = low_sigma - high_sigma
print(dog.shape)
# cv2.imwrite("./blur.jpg", dog)

matrix_r = np.ones_like(dog) * 0.5
