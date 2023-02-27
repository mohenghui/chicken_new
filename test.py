from torch.autograd import Variable
from PIL import Image
import torch
from torchvision.transforms import transforms
import os
import torch.nn as nn
from models.convnext import convnext_base, convnext_tiny
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMCLASS=2
INPUT_CHANNEL=1
INPUT_SIZE=224
font_class=""
sorted_txt="sortfont.txt"
for line in open(sorted_txt, encoding='utf-8'):
    font_class+=line.strip()
transform_test=transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])


model_ft=convnext_base(pretrained=True)
model_input=model_ft.downsample_layers[0][0]

model_ft.downsample_layers[0][0]=nn.Sequential(nn.Conv2d(INPUT_CHANNEL,3,1,1),
                                model_input)

# for k,v in model_input.named_modules():   
#     print(k,"-",v)
#修改最后一层,in_features得到该层的输入
num_ftrs=model_ft.head.in_features
model_ft.head=nn.Linear(num_ftrs,NUMCLASS)


model_ft.load_state_dict(torch.load("model_last_.pth"))
model_ft.eval()
model_ft.to(DEVICE)

test_path="D:\\vscode_work\\yolov7-main\\results"
testList=os.listdir(test_path)
need_check=[]
save_path="./detect_auto"
for list in testList:
    dir_path=os.path.join(test_path,list)
    for file in os.listdir(dir_path):
        img=Image.open(os.path.join(dir_path,file))
        img=transform_test(img)
        img.unsqueeze_(0)
        img=Variable(img).to(DEVICE)
        out=model_ft(img)
        _,pred=torch.max(out.data,1)
        confident_translation=torch.nn.Softmax(dim=1)
        confident,idx=torch.sort(confident_translation(out))
        # print(confident[0][:5]*100)
        print(confident[0]*100)
        # print([font_class[i.item()] for i in idx[0][:5]])
        pred_label=font_class[pred.data.item()]
        print("Image Name:{},predice:{}".format(file,pred_label))
        img_label=file.split('.')[0]
        old_path=os.path.join(dir_path,file)
        new_path=os.path.join(save_path,str(pred.data.item()),file)
        os.rename(old_path,new_path)
        if img_label!=pred_label:
            need_check.append(img_label)
    # print("正确率为",((NUMCLASS-len(need_check))/NUMCLASS)*100)
    # print("请检查"+str(need_check)+"是否出错")
    # dataset_test=FontData(test_path,transform_test,test=True)
    # print(len(dataset_test))

    # for index,(img,label) in enumerate(dataset_test):
    #     img.unsqueeze_(0)
    #     data=Variable(img).to(DEVICE)
    #     output=model(data)
    #     _,pred=torch.max(output.data,1)
    #     print('Image Name:{},predict:{}'.format(dataset_test.imgs[index], classes[pred.data.item()]))