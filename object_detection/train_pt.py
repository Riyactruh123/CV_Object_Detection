from ultralytics import YOLO
import multiprocessing
import torch
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda:0'  # Use the first GPU
    else:
        print("No GPU available, using CPU instead.")
        device = 'cpu'


    # Build a YOLOv9c model from scratch
    # model = YOLO("yolov9c.yaml")

    # Build a YOLOv9c model from pretrained weight
    model = YOLO("yolov9c.pt")
    model = model.to(device)

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="/Users/kushireddy/Desktop/aerial_images/data.yaml",  batch=8, epochs=10,  name='yolov5_e100_shoe_detection', lr0 =0.001, save=True, save_period=5 )
