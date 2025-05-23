# === RUN INFERENCE ===
import cv2
import onnxruntime as ort
import numpy as np

# === CONFIG ===
IMAGE_PATH = "/Users/kushireddy/Downloads/ships-aerial-images/test/images/0cd8c51dd_jpg.rf.17db72d1f449d77200cb82e6b2b5fb9d.jpg"
MODEL_PATH = "/Users/kushireddy/Downloads/ships-aerial-images/best.onnx"
OUTPUT_IMAGE_PATH = "output.jpg"
INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.5
class_names = ["ship"]

# === LOAD IMAGE ===
image = cv2.imread(IMAGE_PATH)
original_image = image.copy()
original_h, original_w = image.shape[:2]

# === PREPROCESS ===
resized_image = cv2.resize(image, INPUT_SIZE)
input_tensor = resized_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
input_tensor = np.expand_dims(input_tensor / 255.0, axis=0).astype(np.float32)

# === RUN INFERENCE ===
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})
outputs = np.squeeze(outputs[0])

detections = outputs.transpose(1, 0)  # [num_detections, 5 + num_classes]

# === PARSE DETECTIONS ===
for pred in detections:
    x, y, w, h = pred[:4]
    class_scores = pred[4:]  # Take only 8 class scores

    class_id = np.argmax(class_scores)
    conf = class_scores[class_id]

    if conf > CONFIDENCE_THRESHOLD:
        x1 = int((x - w / 2) * original_w / 640)
        y1 = int((y - h / 2) * original_h / 640)
        x2 = int((x + w / 2) * original_w / 640)
        y2 = int((y + h / 2) * original_h / 640)

        label = f"{class_names[class_id]} {conf:.2f}"
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            original_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

cv2.imshow("Detection", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()