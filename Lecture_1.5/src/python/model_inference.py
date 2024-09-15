import torch
import numpy as np
import cv2

model = None

def init_model():
	model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

	model = model.to('cpu')

	return model


def inference(image_path):
	if (model is None):
		model = init_model() # Loading a model
  
  	model.eval()
  
	# Load image using OpenCV
	image = cv2.imread(image_path)
	if image is None:
		raise ValueError(f"Failed to load image: {image_path}")

	# Perform inference
    results = model(image)
    
    # Get bounding boxes, confidences, and class labels
    predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
    
    print("PY:", predictions)
    # Prepare results as a numpy array to return
    return predictions