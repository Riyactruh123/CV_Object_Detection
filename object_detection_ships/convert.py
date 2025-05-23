from ultralytics import YOLO

#load a model
model = YOLO('/Users/kushireddy/Downloads/ships-aerial-images/best (2).pt')

#Export the model
model.export(format= 'onnx')  