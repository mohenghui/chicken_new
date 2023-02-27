import sys
from PyQt5.QtCore import QPropertyAnimation, Qt, QRectF
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import QPushButton, QApplication, QStyleOptionButton, \
    QStylePainter, QStyle,QHBoxLayout,QWidget,QLineEdit,QVBoxLayout,QLabel,QFormLayout,QCheckBox

import math
StyleSheet = """
/*这里是通用设置，所有按钮都有效，不过后面的可以覆盖这个*/
QPushButton {
    border: none; /*去掉边框*/
}

/*
QPushButton#xxx
或者
#xx
都表示通过设置的objectName来指定
*/
QPushButton#RedButton {
    background-color: #f44336; /*背景颜色*/
}
#RedButton:hover {
    background-color: #e57373; /*鼠标悬停时背景颜色*/
}
/*注意pressed一定要放在hover的后面，否则没有效果*/
#RedButton:pressed {
    background-color: #ffcdd2; /*鼠标按下不放时背景颜色*/
}

#GreenButton {
    background-color: #4caf50;
    border-radius: 5px; /*圆角*/
}
#GreenButton:hover {
    background-color: #81c784;
}
#GreenButton:pressed {
    background-color: #c8e6c9;
}

#BlueButton {
    background-color: #2196f3;
    /*限制最小最大尺寸*/
    min-width: 96px;
    max-width: 96px;
    min-height: 96px;
    max-height: 96px;
    border-radius: 48px; /*圆形*/
}
#BlueButton:hover {
    background-color: #64b5f6;
}
#BlueButton:pressed {
    background-color: #bbdefb;
}

#OrangeButton {
    max-height: 48px;
    border-top-right-radius: 20px; /*右上角圆角*/
    border-bottom-left-radius: 20px; /*左下角圆角*/
    background-color: #ff9800;
}
#OrangeButton:hover {
    background-color: #ffb74d;
}
#OrangeButton:pressed {
    background-color: #ffe0b2;
}

/*根据文字内容来区分按钮,同理还可以根据其它属性来区分*/
QPushButton[text="purple button"] {
    color: white; /*文字颜色*/
    background-color: #9c27b0;
}
"""


class Window(QWidget):
    def __init__(self, *args,**kwargs) -> None:
        super(Window,self).__init__(*args,**kwargs)
        self.resize(300,600)
        self.setWindowTitle("卷积尺寸计算器") 
        self.layout = QVBoxLayout(self)
        # self.layout = QHBoxLayout(self)


        

        #触发按钮
        self.button1 = QPushButton("计算卷积后的尺寸", self,
                                     objectName="RedButton", minimumHeight=48)
        
        #文本提示
        self.layout1 = QFormLayout()
        self.label1=QLabel("请输入stride值和padding值",self)
        self.label1.setAlignment(Qt.AlignCenter)

        # self.label1.setStyleSheet('text-align:center')

        self.label2=QLabel("stride数值",self)
        self.label2.setAlignment(Qt.AlignCenter)

        self.edit1 = QLineEdit(self) #创建对象


        self.layout1.addRow(self.label2,self.edit1)


        self.layout2 = QFormLayout()
        self.label3 = QLabel("padding数值",self)
        self.label3.setAlignment(Qt.AlignCenter)
        #编辑框

        self.edit2 = QLineEdit(self) #创建对象


        self.layout2.addRow(self.label3,self.edit2)

        # self.edit1.resize(200,50) #自定义大小


        self.layout3 = QFormLayout()
        self.label6 = QLabel("image_size数值",self)
        self.label6.setAlignment(Qt.AlignCenter)
        #编辑框

        self.edit3 = QLineEdit(self) #创建对象


        self.layout3.addRow(self.label6,self.edit3)


        self.layout4 = QFormLayout()
        self.label8 = QLabel("kernel_size数值",self)
        self.label8.setAlignment(Qt.AlignCenter)
        #编辑框

        self.edit4 = QLineEdit(self) #创建对象


        self.layout4.addRow(self.label8,self.edit4)


        self.layout5 = QHBoxLayout()
        self.check1=QCheckBox("same",self)
        self.check1.setProperty("id",1)

        self.check2=QCheckBox("vaild",self)
        self.check2.setProperty("id",2)

        self.check3=QCheckBox("None",self)
        self.check3.setProperty("id",3)

        # self.check3.setTristate(True)
        self.check3.setCheckState(Qt.Checked)

        self.check1.stateChanged.connect(self.changeCheck1)
        self.check2.stateChanged.connect(self.changeCheck2)
        self.check3.stateChanged.connect(self.changeCheck3)


        self.layout5.addWidget(self.check1)
        self.layout5.addWidget(self.check2)
        self.layout5.addWidget(self.check3)
        



        self.label4=QLabel("常规标准卷积计算公式为:((image_size+2*padding-kernel_size)/stride)+1 向下取整",self)
        self.label4.setAlignment(Qt.AlignCenter)
        
        self.label6=QLabel("padding为SAME卷积计算公式为:(image_size)/stride 向上取整",self)
        self.label6.setAlignment(Qt.AlignCenter)

        self.label7=QLabel("padding为VALID卷积计算公式为:(image_size-kernel_size+1)/stride 向上取整",self)
        self.label7.setAlignment(Qt.AlignCenter)


        self.label5=QLabel("答案为:",self)
        self.label5.setAlignment(Qt.AlignCenter)
        

        self.layout.addWidget(self.label1)
        self.layout.addLayout(self.layout3)        
        self.layout.addLayout(self.layout2)
        self.layout.addLayout(self.layout4)
        self.layout.addLayout(self.layout1)
        self.layout.addLayout(self.layout5)

        self.layout.addWidget(self.label4)        
        self.layout.addWidget(self.label6)
        self.layout.addWidget(self.label7)
        self.layout.addWidget(self.button1)
        self.layout.addWidget(self.label5)
        # self.layout.setAlignment(Qt.AlignCenter)
        # self.edit1.move(0,0)  #自定义位置

        self.bind_trigger()


        
        # 设置垂直盒布局的控件间距大小
        # self.layout.setSpacing(20)
        # self.setLayout(self.layout)

    def  changeCheck1(self):
        if self.check1.checkState() == Qt.Checked:
            self.check2.setChecked(False)
            self.check3.setChecked(False)
    def  changeCheck2(self):
        if self.check2.checkState() == Qt.Checked:
            self.check1.setChecked(False)
            self.check3.setChecked(False)
    def  changeCheck3(self):
        if self.check3.checkState() == Qt.Checked:
            self.check1.setChecked(False)
            self.check2.setChecked(False)

    def bind_trigger(self):
        self.button1.clicked.connect(self.conv_cal)

    def conv_cal(self):
        ans=0
        image_size=int(self.edit3.text()) if self.edit3.text() else None
        stride=int(self.edit1.text()) if self.edit1.text() else None
        padding=int(self.edit2.text()) if self.edit2.text() else None
        kernel_size=int(self.edit4.text()) if self.edit4.text() else None
        if self.check1.isChecked():
            ans=int(math.ceil(image_size/stride))
            self.label_set(image_size,ans)

        elif self.check2.isChecked():
            ans=int(math.ceil((image_size-kernel_size+1)/stride))
            self.label_set(image_size,ans)
        elif self.check3.isChecked():
            ans=int(math.floor((image_size+2*padding-kernel_size)/stride)+1)
            self.label_set(image_size,ans)
    def label_set(self,input,ouput):
        self.label5.setText("卷积前卷积为{}卷积后的尺寸为{}".format(str(input),str(ouput)))
        # self.label5.setText("<b style='color:red'>Button has been clicked!</b>")

if __name__=="__main__":
    app=QApplication(sys.argv)
    QFontDatabase.addApplicationFont(        
        "Data/Fonts/FontAwesome/fontawesome-webfont.ttf")
    app.setStyleSheet(StyleSheet)
    w=Window()
    w.show()
    sys.exit(app.exec_())