from cProfile import label
import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils import data
Labels={"female":0,"male":1}
class FontData(data.Dataset):
    def getDate(self,root):
        imgs=[]
        imgs_labels=[os.path.join(root,img) for img in os.listdir(root)]
        for imglabel in imgs_labels:
            for imgname in os.listdir(imglabel):
                imgpath=os.path.join(imglabel,imgname)
                imgs.append(imgpath)
        return imgs
    def __init__(self,root,transforms=None,train=True,test=False) -> None:
        super().__init__()
        self.test=test
        self.transforms=transforms

        if self.test: #test文件夹只有一层 root:数据集/test
            imgs=[os.path.join(root,img) for img in os.listdir(root)]#获得图片文件路劲
            self.imgs=imgs
        else:#root:数据集/train
            # imgs=[] #train文件夹第一层为标签
            # trainval_files,val_files=train_test_split(imgs,test_size=0.3,random_state=42)
            if train:
                self.imgs=self.getDate(root)
            else:
                self.imgs=self.getDate(root)
    def __getitem__(self, index) :
        img_path=self.imgs[index]
        img_path=img_path.replace("\\",'/')
        # img_path=Path(img_path)
        if self.test:
            label=-1 #没有标签
        else:
            labelname=(Labels[img_path.split('/')[-2]])
            label=labelname
        data=Image.open(img_path)
        data=self.transforms(data)
        return data,label
    def __len__(self):
        return len(self.imgs)
