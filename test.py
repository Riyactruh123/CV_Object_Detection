from ultralytics import YOLO
import torch
import multiprocessing
import os
import cv2
import shutil

# === CONFIGURATION ===
MODEL_PATH = "/Users/kushireddy/Desktop/aerial_images/best_images.pt"
INPUT_FOLDER = "/Users/kushireddy/Desktop/aerial_images/test/images"
OUTPUT_FOLDER = "output_images"
CONFIDENCE_THRESHOLD = 0.25
IMG_SIZE = 640

# Define traffic level function
def classify_traffic(count):
    if count < 5:
        return "Low"
    elif 5 <= count <= 15:
        return "Moderate"
    else:
        return "Huge"

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
    model = YOLO(MODEL_PATH).to(device)

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Loop through each image in the input folder
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Could not read image {filename}")
            continue

        # Run inference
        result = model.predict(
            source=image_path,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=device,
            verbose=False
        )[0]

        detections = result.boxes
        class_names = model.names  # Get class name mapping
        count = 0

        # Draw detections
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = class_names[cls] if cls in class_names else f"Class {cls}"

            count += 1
            label = f"{class_name} ({conf:.2f})"
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Traffic label
        traffic_level = classify_traffic(count)
        summary = f"Vehicles: {count} | Traffic: {traffic_level}"
        cv2.putText(image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        # Save result
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, image)
        print(f"[INFO] Processed {filename} | {summary}")

    print(f"[INFO] All images saved in: {OUTPUT_FOLDER}")
