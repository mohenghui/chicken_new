from cProfile import label
from importlib.resources import path
import cv2
# from tkinter import Image, Variable
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision.transforms import transforms
import os
import torch.nn as nn
from dataset.dataset import FontData
from models.convnext import convnext_base, convnext_tiny
# from train import DEVICE, INPUT_SIZE
def init_model(self):
    self.DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.NUMCLASS=2
    self.INPUT_CHANNEL=1
    self.INPUT_SIZE=224
    self.transform_test=transforms.Compose([
        transforms.Resize((self.INPUT_SIZE,self.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    self.model_ft=convnext_base(pretrained=True)
    model_input=self.downsample_layers[0][0]

    self.model_ft.downsample_layers[0][0]=nn.Sequential(nn.Conv2d(self.INPUT_CHANNEL,3,1,1),
                                    model_input)
    num_ftrs=self.model_ft.head.in_features
    self.model_ft.head=nn.Sequential(nn.Linear(num_ftrs,self.NUMCLASS),
    nn.Softmax(1))#修改成对应层数
    self.model_ft.load_state_dict(torch.load("model_last_.pth"))
    self.model_ft.eval()
    self.model_ft.to(self.DEVICE)
    label_class=["母鸡","公鸡"]
def predict(self,img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=self.transform_test(img)
    img.unsqueeze_(0)
    img=Variable(img).to(self.DEVICE)
    out=self.model_ft(img)
    _,pred=torch.max(out.data,1)
    confident_translation=torch.nn.Softmax(dim=1)
    pred_label=self.label_class[pred.data.item()]
    print("predice:{}".format(pred_label))
    return pred_label
