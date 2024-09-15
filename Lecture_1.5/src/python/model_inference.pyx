import torch
import numpy as np
import cv2
import sys  # Import sys to flush the output

cdef object model = None

def init_model():
    global model
    print("Initializing the model...")  # Debugging
    sys.stdout.flush()  # Force flush the output
    if model is None:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model = model.to('cpu')
    return model

def inference(image_path):
    global model
    print("Inference function called.")  # Debugging
    sys.stdout.flush()  # Force flush the output

    if model is None:
        print("Model is not initialized. Initializing now...")  # Debugging
        sys.stdout.flush()  # Force flush the output
        model = init_model()

    model.eval()

    # Convert the input to a bytes object if it is not already
    if isinstance(image_path, str):
        image_path = image_path.encode('utf-8')

    # Load image using OpenCV
    image = cv2.imread(image_path.decode('utf-8'))  # Convert bytes to string
    if image is None:
        print(f"Failed to load image: {image_path.decode('utf-8')}")
        sys.stdout.flush()  # Force flush the output
        return np.array([])  # Return an empty array to avoid crashes in the C++ code

    print("Image loaded successfully.")  # Debugging
    sys.stdout.flush()  # Force flush the output

    # Perform inference
    results = model(image)
    
    # Get bounding boxes, confidences, and class labels
    predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    print(f"Predictions: {predictions}")  # Debugging
    sys.stdout.flush()  # Force flush the output
    
    return predictions


