import os
import shutil
import random
# data_dir="./detect_auto/1"
data_dir="./data/train/male"
# new_dir="./cuo/1"
new_dir="./data/valid/male"
get_num=800
file_list=random.shufle(os.listdir(data_dir))
n=len(file_list)
skip=n//get_num
index=0
for file in (file_list):
    # if (idx+1)%skip
    if index==skip:
        old_name=os.path.join(data_dir,file)
        new_name=os.path.join(new_dir,file)
        # shutil.copy(old_name,new_name)
        os.rename(old_name,new_name)
        index=0
    index+=1

