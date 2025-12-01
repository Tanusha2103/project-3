#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:46:02 2025

@author: tanu
"""

# imports
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO


#-------------Step 1: Object Masking-------------------

BASE = "/Users/tanu/Desktop/project 3/Project 3 data"
DATA_YAML = os.path.join(BASE, "data", "data.yaml")
eval_dir = os.path.join(BASE, "data", "evaluation")
IMG_MOTHERBOARD = os.path.join(BASE, "motherboard_image.JPEG")


#Load image
img_bgr = cv2.imread(IMG_MOTHERBOARD)

# Rotate only if needed
h, w = img_bgr.shape[:2]
if h > w:  # portrait -> rotate to landscape
    img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    
# Copy of the original for final masking
img_orig = img_bgr.copy()

# Convert to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Blur to remove noise but preserve edges
gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Invert dark
_, binary = cv2.threshold(
    gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

edges = cv2.Canny(closed, 60, 180)

contours, _ = cv2.findContours(
    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
main_contour = max(contours, key=cv2.contourArea)

mask = np.zeros_like(gray)             
cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

# Apply mask to original colour image
extracted_bgr = cv2.bitwise_and(img_orig, img_orig, mask=mask)
extracted_rgb = cv2.cvtColor(extracted_bgr, cv2.COLOR_BGR2RGB)

# For plotting
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

# Plots
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(orig_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(edges_rgb)
plt.title("Edges (Canny)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(mask_rgb)
plt.title("Binary Mask")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(extracted_rgb)
plt.title("Extracted Motherboard")
plt.axis("off")

plt.tight_layout()
plt.show()

#-------------Step 2: YOLOv11 Training-------------------

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load a pretrained YOLOv11-nano backbone
yolo_model = YOLO("yolo11n.pt") 

results = yolo_model.train(
    data=DATA_YAML,
    epochs=3,          
    batch=2,             
    imgsz=900,          
    name="pcb_yolo11n6",  
    project="runs",   
    device="mps",
    val=False
)
                    #########
BEST = "/Users/tanu/runs/pcb_yolo11n6/weights/best.pt"
model = YOLO(BEST)

val_results = model.val(
    data=DATA_YAML,
    imgsz=900,
    device="mps",
    plots=True,   # <-- this generates all graphs
    save_json=False,
    project="/Users/tanu/runs",
    name="pcb_yolo11n6_val"
)

val_dir = val_results.save_dir
print("VAL DIR:", val_dir)
print(os.listdir(val_dir))

import shutil

DEST = "/Users/tanu/Desktop/project 3/Step2_results"

os.makedirs(DEST, exist_ok=True)

for file in os.listdir(val_dir):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        src = os.path.join(val_dir, file)
        dst = os.path.join(DEST, file)
        shutil.copy(src, dst)
        print("Copied:", file)

                    ########
run_dir = results.save_dir

fig_files = {
    "Confusion matrix (normalized)": "confusion_matrix_normalized.png",
    "Precision–Confidence curve":    "P_curve.png",
    "Precision–Recall curve":        "PR_curve.png"
}

for title, fname in fig_files.items():
    path = os.path.join(run_dir, fname)
    if os.path.exists(path):
        img = Image.open(path)
        plt.figure(figsize=(6, 5))
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
#-------------Step 3: Evaluation-------------------

weights_path = os.path.join(run_dir, "weights", "best.pt")
model = YOLO(weights_path)

eval_images = [
    os.path.join(eval_dir, "ardmega.jpg"),
    os.path.join(eval_dir, "arduno.jpg"),
    os.path.join(eval_dir, "rasppi.jpg")
]

results = model.predict(
    source=eval_images,
    imgsz=1024,          
    conf=0.25,           # confidence threshold
    name="pcb_eval",
    project="runs",
    save=True,
    device="mps"               # saves annotated images automatically
)

out_dir = results[0].save_dir  
print("Annotated images saved in:", out_dir)

for img_name in os.listdir(out_dir):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(out_dir, img_name)
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 4))
        plt.imshow(rgb)
        plt.title(f"Detection result: {img_name}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
