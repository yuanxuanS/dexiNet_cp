import cv2
import os

def canny_an_img(img_p, save_p):
    # img_p = "/home/panpan/DexiNed/datasets/BIPEDv2/BIPEDv2/BIPED/edges/imgs/train/rgbr/real/RGB_001.jpg"

    img = cv2.imread(img_p, cv2.IMREAD_COLOR)
    print(f"image shape", img.shape)
    canny_ = cv2.Canny(img, 100, 200)       # 非极大阈值的 两个阈值
    cv2.imwrite(save_p, canny_)
    
def canny_imgs(img_dir, save_dir):
    all_img = os.walk(img_dir)
    
    for root, dirs, img_p in all_img:
        # print(img_p)
        for img in img_p:
            file_ext = os.path.splitext(img)
            
            front, ext = file_ext
            save_p = save_dir + '/' + front + '_canny.jpg'
            canny_an_img(img_dir + "/" + img, save_p)
        # save_p = save_dir + "/" + 
'''
img_dir = "/home/panpan/DexiNed/datasets/BIPEDv2/BIPEDv2/BIPED/edges/imgs/test/rgbr/RGB_008.jpg"
save_canny_dir = "/home/panpan/DexiNed/tmp.jpg"
# canny_imgs(img_dir, save_canny_dir)
canny_an_img(img_dir, save_canny_dir)

'''
path = "./blur.jpg"
resize_path = "./blur_resize.jpg"
save_p = "./blur_canny.jpg"
img = cv2.imread(path, cv2.IMREAD_COLOR)
w, h = 1280, 720
img_resize = cv2.resize(img, (w, h))
cv2.imwrite(resize_path, img_resize)
print(img_resize.shape)
canny_an_img(resize_path, save_p)