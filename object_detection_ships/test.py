from ultralytics import YOLO
import torch
import multiprocessing
import os
import shutil

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda:0' 
    else:
        print("No GPU available, using CPU instead.")
        device = 'cpu'

    # Load the trained YOLO model
    model = YOLO("/Users/kushireddy/Downloads/ships-aerial-images/best (2).pt")
    model = model.to(device)

    # Define the folder containing test images
    input_folder = "/Users/kushireddy/Downloads/ships-aerial-images/test/images"  
    output_folder = "output_images/" 

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Run inference on all images in the folder and save them
    results = model.predict(
        source=input_folder,  
        save=True,            
        save_txt=True,        
        save_conf=True,       
        imgsz=640,           
        conf=0.25,            
        device=device        
    )

    # Move saved images to the output folder
    pred_folder = "runs/detect/predict" 
    for filename in os.listdir(pred_folder):
        source_path = os.path.join(pred_folder, filename)
        dest_path = os.path.join(output_folder, filename)
        shutil.move(source_path, dest_path)

    print(f"Processed images saved in: {output_folder}")  