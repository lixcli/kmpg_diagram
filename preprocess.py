"""
将raw_data 和 raw_label 分成train test val文件夹
"""

from sklearn.model_selection import train_test_split
import os
import shutil
from tqdm import tqdm
test_size = 0.2
val_size = 0.1
# raw_data_pth = "./raw_data"
raw_label_pth = "./raw_label"
raw_data_pth = "./raw_data"
dataset = "./VOCdevkit/VOC2019"

if __name__ == "__main__":

    raw_label_list=[]
    for root, dirs, files in os.walk(raw_label_pth):
        for file in files:
            # if "(" not in files and ")" not in files:
            # file.replace(' ','-')
            if "DS" in files:
                continue
            shutil.copy(f"{raw_label_pth}/{file}",f"{dataset}/Annotations/{file}")
            # shutil.move(f"{raw_data_pth}/{file[:-4]}.jpg",f"{dataset}/JEPGImages/{file[:-4]}.jpg")
            raw_label_list.append(file)
    
    
    # raw_data_list = [x[:-4]+".jpg" for x in raw_label_list]

    train_val_label, test_label = train_test_split(
                                                raw_label_list,
                                                test_size=test_size,
    )
    train_label, val_label = train_test_split(
                                                train_val_label,
                                                test_size=val_size,
    )

    print("move train label")
    with open(f"{dataset}/ImageSets/Main/train.txt","a") as f:
        for label in tqdm(train_label):
            # shutil.copy(f"{raw_label_pth}/{label}",f"{dataset}/train/{label}")
            f.write(f"{label[:-4]}\n")
    print("move test label")
    with open(f"{dataset}/ImageSets/Main/test.txt","a") as f:
        for label in tqdm(test_label):
            # shutil.copy(f"{raw_label_pth}/{label}",f"{dataset}/test/{label}")
            f.write(f"{label[:-4]}\n")

    print("move val label")
    with open(f"{dataset}/ImageSets/Main/val.txt","a") as f:
        for label in tqdm(val_label):
            shutil.copy(f"{raw_label_pth}/{label}",f"testImg/{label}")
            f.write(f"{label[:-4]}\n")



