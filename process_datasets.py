import pandas as pd

# data = pd.read_csv("/home/panpan/DexiNed/datasets/DUTS/metadata.csv")
# is_test = data["split"] == "test"

# test_data = data[is_test]
# train_data = data[~is_test]
'''
train_lst = []
file = open("train.lst", "w")
for row_index, row in train_data.iterrows():
    lst = []
    # lst.append(row["image_path"])
    # lst.append(row["mask_path"])
    file.write(row["image_path"]+" "+row["mask_path"])
    file.write("\r\n")
file.close()
'''
import os

image_directories = "/home/panpan/DexiNed/datasets/BIPEDv2/BIPEDv2/BIPED_pred"
file = open(image_directories+"/test_pair.lst", "w")

for file_name_ext in os.listdir(image_directories+"/imgs"):
    # file_name = os.path.splitext(file_name_ext)[0]
    file_name_ext = "imgs/"+file_name_ext
    file.write(file_name_ext+" "+file_name_ext)
    file.write("\r\n")
file.close()