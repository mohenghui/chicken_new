from cmath import inf
from math import cos, sin
import numpy as np
import math
from dis import dis
import os
import platform
import base64
import json
from PIL import Image, ImageDraw, ImageFont
import cv2
IMG_TAIL = ".png"
#将返回的错误码转换为十六进制显示
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2**32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr   
    return hexStr

def makedirR(c_path, is_dir=True):
    if is_dir and not os.path.exists(c_path):
        os.mkdir(c_path)
    elif not is_dir and not os.path.exists(c_path):  # 文件新建上一级目录
        if platform.system().lower() == 'windows':
            tmp = '\\'.join(c_path.split('\\')[:-1])
        elif platform.system().lower() == 'linux':
            tmp = '/'.join(c_path.split('/')[:-1])
        if not os.path.exists(tmp):
            os.mkdir(tmp)


def imagedecode(j_path, save_path):
    with open(j_path, "r") as json_file:
        raw_data = json.load(json_file)
    image_base64_string = raw_data["imageData"]
    # 将 base64 字符串解码成图片字节码
    image_data = base64.b64decode(image_base64_string)
    # 将字节码以二进制形式存入图片文件中，注意 'wb'
    file_path, file_name = os.path.split(j_path)
    save_path = os.path.join(
        save_path, os.path.splitext(file_name)[0]+IMG_TAIL)
    with open(save_path, 'wb') as jpg_file:
        jpg_file.write(image_data)

def beint(tuple_list):
    return tuple([int(i) for i in tuple_list])
def listint(list):
    return [int(i)for i in list]
def cal_point(point1,point2):
    
    x1=point1[0][0]#取四点坐标
    y1=point1[0][1]
    x2=point1[1][0]
    y2=point1[1][1]
    
    x3=point2[0][0]
    y3=point2[0][1]
    x4=point2[1][0]
    y4=point2[1][1]
    
    k1=(y2-y1)*1.0/(x2-x1)#计算k1,由于点均为整数，需要进行浮点数转化
    b1=y1*1.0-x1*k1*1.0#整型转浮点型是关键
    if (x4-x3)==0:#L2直线斜率不存在操作
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)#斜率存在操作
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0
    return [x,y]
def cal_distance(point1,point2):
    return pow(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2),1/2)


def get_foot(start_point, end_point, point_a):
    start_x, start_y = start_point
    end_x, end_y = end_point
    pa_x, pa_y = point_a

    p_foot = [0, 0]
    if start_point[0] == end_point[0]:
        p_foot[0] = start_point[0]
        p_foot[1] = point_a[1]
        return p_foot

    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot[0] = int((b * b * pa_x - a * b * pa_y - a * c) / (a * a + b * b))
    p_foot[1] = int((a * a * pa_y - a * b * pa_x - b * c) / (a * a + b * b))

    return p_foot
def sign_cal_distance(point1,point2):
    if point1[1]<point2[1]:
        return pow(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2),1/2)
    else:
        return -pow(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2),1/2)

def scale_distance(distance1,distance2,scale):
    return True if distance1*scale<=distance2 or distance2*scale<=distance1 else False


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def cal_scale(line1,line2,scale):
    return True if line1>=line2*scale or line2>=line1*scale else False

def get_angle(line1, line2):
    """
    计算两条线段之间的夹角
    :param line1:
    :param line2:
    :return:
    """
    dx1 = line1[0][0] - line1[1][0]
    dy1 = line1[0][1] - line1[1][1]
    dx2 = line2[0][0] - line2[1][0]
    dy2 = line2[0][1] - line2[1][1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        insideAngle = abs(angle1 - angle2)
    else:
        insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle = insideAngle % 180
    return insideAngle
def beint(tuple_list):
    return tuple([int(i) for i in tuple_list])
def listint(list):
    return [int(i)for i in list]
def listint_mul(lists):
    result=[]
    for tmp_list in lists:
        target_list=[int(i)for i in tmp_list ]
        result.append(target_list)
    return result
def makedir(c_path, file_flag=False):
    if file_flag:
        c_path = os.path.dirname(c_path)
    if not os.path.exists(c_path):
        father_dir = os.path.dirname(c_path)
        if not os.path.exists(father_dir):
            makedir(father_dir)
        os.mkdir(c_path)
def show_chinese(img,text,pos):
    """
    :param img: opencv 图片
    :param text: 显示的中文字体
    :param pos: 显示位置
    :return:    带有字体的显示图片（包含中文）
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font='msyh.ttc', size=36)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(255, 0, 0))  # PIL中RGB=(255,0,0)表示红色
    img_cv = np.array(img_pil)                         # PIL图片转换为numpy
    img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)      # PIL格式转换为OpenCV的BGR格式
    return img
def chose_licence_plate(contours, Min_Area=0,Max_Area=inf):
    # 根据侧拍的物理特征对所得矩形进行过滤
    # 输入：contours每个轮廓列表的特征是一个三维数组 N*1*2
    # 输出：返回经过过滤后的轮廓集合
    temp_contours = []
    for contour in contours:
        # 对符合面积要求的巨型装进list
        contoursize=cv2.contourArea(contour)
        if contoursize > Min_Area and contoursize < Max_Area:
            # print("矩形框大小",contoursize)
            temp_contours.append(contour)
    # car_plate = []
    # for item in temp_contours:
    #     # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
    #     rect = cv2.boundingRect(item)
    #     x = rect[0]
    #     y = rect[1]
    #     weight = rect[2]
    #     height = rect[3]
    #     # 440mm×140mm
    #     # print("宽高比：",weight/height)
    #     # if (weight > (height * 0.45)) and (weight < (height * 1.5)):
    #     car_plate.append(item)
    # 返回车牌列表
    return temp_contours

def draw(list,x0,y0):
    newx=0
    newy=0
    weight=0
    height=0
    result=[]
    for item in list:
        # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        newx=x+x0
        newy=y+y0
        # print("新%sx,新%sy"%(newx,newy))
        result.append([newx,newy,newx+weight,newy+height])
    return result
def cal_distance(point1,point2):
    return pow(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2),1/2)

def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
 
    M = cv2.getRotationMatrix2D(center, angle, scale) #5
 
    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7

def contours_area(cnt):
    # 计算countour的面积
    (x, y, w, h) = cv2.boundingRect(cnt)
    return w * h