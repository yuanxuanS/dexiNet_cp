import cv2
import os
# for k in range(4):
#     p = "./"+str(k)+".jpg"
#     cv2.imread(p, cv2.IMREAD_COLOR)
    

def crop_save_an_img(img_name, path, patch_save_dir, resize_save_dir):
    '''
    把图像分割为2*2，然后保存
    '''
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w= img.shape[0], img.shape[1]

    patch_h, patch_w = int(h / 2), int(w / 2)
    # print(patch_h, patch_w)

    count = 0
    xw_start = 0
    for i in range(2):
        yh_start = 0 
        for j in range(2):
            img_patch = img[yh_start:yh_start+patch_h, xw_start:xw_start+patch_w]       # row范围是opencv图像中的y轴
            print(f"from {xw_start},{yh_start} to {xw_start+patch_w},{yh_start+patch_h}")
            cv2.imwrite(patch_save_dir+"/"+img_name+"_"+str(count)+".jpg", img_patch)
            img_resize = cv2.resize(img_patch, (w, h))
            cv2.imwrite(resize_save_dir+"/"+img_name+"_"+str(count)+".jpg", img_resize)
            yh_start += patch_h
        
            count += 1
        xw_start += patch_w

# print(f"w {w}, h {h}")
# cv2.imwrite("./008.jpg", img)

img_dir = "/home/panpan/DexiNed/result_pp/2024-01-17_16-56-25/BIPED2BIPED_pp/avg"
patch_save_dir = "/home/panpan/DexiNed/result_pp/2024-01-17_16-56-25/BIPED_patch_prededge_origin"
resize_save_dir = "/home/panpan/DexiNed/result_pp/2024-01-17_16-56-25/BIPED_patch_prededge_resize"
for img in os.listdir(img_dir):
    # print(img)
    img_path = os.path.join(img_dir, img)
    img_name = img[:-4]
    crop_save_an_img(img_name, img_path, patch_save_dir, resize_save_dir)

