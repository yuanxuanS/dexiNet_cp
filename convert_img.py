from PIL import Image
import numpy as np

import scipy.io as io
import os

src_img = './canny_img.jpg'
img = Image.open(src_img)
res = np.array(img, dtype='uint16')

file_ext = os.path.splitext(src_img)
front, ext = file_ext       # 分割开的名字和格式
        
io.savemat('./'+front+'.mat', {'result':res})