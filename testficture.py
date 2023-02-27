import cv2
import os 
import numpy as np
from utils import *
from matplotlib import pyplot as plt
root_path="data"
erros_list=[]
#将背景去除的传统方法实现
def bad_mask(path):
    img=cv2.imread(path)
    h,w,c=img.shape
    # print(filepath)
    new_h,new_w=int(h*0.3),int(w*0.3)
    new_img=cv2.resize(img,(new_h,new_w))
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gau_img=cv2.GaussianBlur(gray_img,(3,3),0)
    thresh_img = cv2.threshold(gau_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, None, None, 3)
    # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, None, None, 3)
    # # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, None, None, 3)
    # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, None, None, 3)
    # mor_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, None, None, 3)
    contours, hierarchy = cv2.findContours(mor_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # list = chose_licence_plate(contours)
    # 获取面积最大的contour
    max_cnt = max(contours, key=lambda cnt: contours_area(cnt))
    # 创建空白画布
    mask = np.zeros_like(mor_img)
    # 获取面积最大的 contours
    mask = cv2.drawContours(mask,[max_cnt],0,255,-1)
    sub_img = cv2.bitwise_or(new_img,new_img,mask=mask)
    # 打印罩层
    # plt.imshow(mask, cmap='gray')
    cv2.imshow("mask",mask)
    
    cv2.imshow("sub_img",sub_img)
    # sub_img = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow("og",new_img)
    cv2.imshow("img",mor_img)


# for file in os.listdir(root_path): 
#     dir_path=os.path.join(root_path,file)
#     for f in os.listdir(dir_path):
#         dir_path1=os.path.join(dir_path,f)
#         for ff in os.listdir(dir_path1):
#             filepath=os.path.join(dir_path1,ff)
#             bad_mask()
#             k = cv2.waitKey(0) & 0xFF
#             if k==ord('q'):
#                 exit(0)
#             else:
#                 continue
            # cv2.waitKey(0)
            
# print(erros_list)
root_dir="G:\\mhh\\mydataset\\bad"
for i in os.listdir(root_dir):
    file_path=os.path.join(root_dir,i)
    bad_mask(file_path)
    k = cv2.waitKey(0) & 0xFF
    if k==ord('q'):
        exit(0)
    else:
        continue