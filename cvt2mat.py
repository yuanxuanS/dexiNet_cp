from scipy.io import savemat
import cv2
import os
from PIL import Image
import numpy as np

import scipy.io as io
# convert GT to .mat
# src_dir = "/home/panpan/DexiNed/datasets/BIPEDv2/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/"
src_dir = "/home/panpan/DexiNed/result_pp/2024-01-17_16-56-25/BIPED2BIPED_pp/avg/"
save_dir = "/home/panpan/DexiNed/result_pp/2024-01-17_16-56-25/BIPED2BIPED_pp/avg_mat/"
# save_dir = "/home/panpan/DexiNed/datasets/BIPEDv2/BIPEDv2/BIPED/edges/edge_maps/test/rgbr_mat/"

all_file = os.walk(src_dir)
fileNum = 0
for root, dirs, files in all_file:
    # files: 文件夹下的子文件名的list，无前面的路径
    # dirs: 文件夹下的子文件 夹的list，没有则返回 []
    # print(root, dirs, files)
    for file in files:
        fileNum = fileNum + 1
        print(src_dir+file)
        file_ext = os.path.splitext(file)
        front, ext = file_ext       # 分割开的名字和格式
        
        img = Image.open(src_dir+file)
        #先Image读取的图保存为.npy
        res = np.array(img, dtype='uint16')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(save_dir+front+'.npy', res)
        #再由npy保存为.mat
        numpy_file = np.load(save_dir+front+'.npy')
        io.savemat(save_dir+front+'.mat', {'result':numpy_file})
        
#删除中间文件mat
# 为什么先转化为nupmy？mat要求是数组类型。 Image读取返回的是Image class， 通过numpy将其转化为数组类型
delete_command ='rm -rf '+save_dir+'*.npy'
print(delete_command)
os.system(delete_command)
print('共转化了'+ str(fileNum)+'张')