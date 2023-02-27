import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convnext import convnext_base, convnext_tiny
#自定义网络定义
class Mymodel(nn.Module):
    def __init__(self,input_channel=1,num_classes=3):
        super().__init__()
        self.INPUT_CHANNEL=input_channel
        self.NUMCLASS=num_classes
        model = convnext_base(pretrained=True)
        model_input = model.downsample_layers[0][0]

        model.downsample_layers[0][0] = nn.Sequential(nn.Conv2d(self.INPUT_CHANNEL, 3, 1, 1),
                                                                model_input)
        num_ftrs = model.head.in_features
        model.head = nn.Sequential(nn.Linear(num_ftrs, self.NUMCLASS),
                                            nn.Softmax(1))  # 修改成对应层数
        self.model=model
    def get_model(self):
        return self.model
        