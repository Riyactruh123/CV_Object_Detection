from ultralytics import YOLO

#load a model
model = YOLO('best.pt')

#Export the model
model.export(format= 'onnx')  