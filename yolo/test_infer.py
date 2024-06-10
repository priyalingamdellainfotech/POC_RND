import cv2
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

img = cv2.imread("bus1.jpg")

# Inference
results = model(img)

# Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.pandas()  # or .show(), .save(), .crop(), .pandas(), etc.

results.save('final_te.jpg')