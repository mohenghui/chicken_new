import torch
import torch.nn as nn
from torch.autograd import Variable
import onnx
from onnx import shape_inference
from mymodel import Mymodel
if __name__ == "__main__":
    mymodel=Mymodel(1,3).get_model()
    # print(mymodel)
    mymodel.load_state_dict(torch.load('model_loss_86_0.601.pth',
    map_location=torch.device('cpu')))
    # prepare dummy input (this dummy input shape should same with train input)
    dummy_input = Variable(torch.randn(1, 1, 224, 224))
    # export onnx model
    torch.onnx.export(mymodel, dummy_input, "./output/chicken_model.onnx")
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load("./output/chicken_model.onnx")), "./output/chicken_model.onnx")