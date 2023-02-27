from unittest import skip
import cv2
from PIL import Image
import numpy as np
import os
root_dir="./common"
first_index=0
class_name={
"鬼灭之刃 游郭篇":"39433","鬼灭之刃 无限列车篇":"39444",
"鬼灭之刃":"26801","咒术回战":"34430","间谍过家家":"41410",
"辉夜大小姐想让我告白 -究极浪漫-":"41411","国王排名":"39462",
"工作细胞：细胞大作战":"38939","关于我转生变成史莱姆这档事 第二季":"36170",
"小林家的龙女仆 第二季":"38921","Re：从零开始的异世界生活 第二季 后半":"36429",
"工作细胞 第二季":"36174","关于我转生变成史莱姆这档事 转生史莱姆日记":"38221",
"工作细胞":"24588","紫罗兰永恒花园 剧场版":"40028","小林家的龙女仆":"5800"
}

# for animation_list in os.listdir(root_dir):
#     if 

def get_lei():
    rt_class=[]
    for name in (class_name):
        rt_class.append(class_name[name])
    for name in os.listdir(root_dir):
        if name.split('_')[0] not in rt_class:
            os.remove(os.path.join(root_dir,name))
def sub_bad():
    bad_dir="./data/oimages"
    exit_list=os.listdir(bad_dir)
    for i in os.listdir(root_dir):
        if i in exit_list:
            os.remove(os.path.join(root_dir,i))

sub_bad()