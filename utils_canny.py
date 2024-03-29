import cv2

path = "/home/panpan/DexiNed/datasets/BIPEDv2/BIPEDv2/BIPED/edges/imgs/test/rgbr/RGB_008.jpg"
img = cv2.imread(path, cv2.IMREAD_COLOR)


# Apply 3x3 and 7x7 Gaussian blur
low_sigma = cv2.GaussianBlur(img,(3,3),0)
high_sigma = cv2.GaussianBlur(img,(7,7),0)

# Calculate the DoG by subtracting
dog = low_sigma - high_sigma

cv2.imwrite("./DoG_008_3_7.jpg", dog)