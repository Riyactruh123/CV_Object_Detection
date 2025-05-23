from ultralytics import YOLO
import multiprocessing
import torch

# Check if GPU is available and set the device
if __name__ == '__main__':
    multiprocessing.freeze_support()

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda:0'  # Use the first GPU
    else:
        print("No GPU available, using CPU instead.")
        device = 'cpu'

    # Build a YOLOv9c model from scratch (optional)
    # model = YOLO("yolov9c.yaml")

    # Build a YOLOv9c model from a pretrained weight
    model = YOLO("yolov8n.pt")
    model = model.to(device)

    # Display model information (optional)
    model.info()

    # Train the model
    # Ensure the path to data.yaml is correct and the file exists at this location.
    # Replace "/Users/kushireddy/Desktop/aerial_images/data.yaml" with the actual path if it's different.
    data_yaml_path = "/content/drive/MyDrive/ships-aerial-images/data.yaml"

    try:
        results = model.train(
            data=data_yaml_path,
            batch=16,
            epochs=50,
            name='yolov8_50e_objectdectection',
            lr0=0.001,
            save=True,
            save_period=5
        )
    except FileNotFoundError:
        print(f"Error: The file '{data_yaml_path}' was not found.")
        print("Please verify the path to your data.yaml file and ensure it exists.")
    except Exception as e:
        print(f"An error occurred during training: {e}")