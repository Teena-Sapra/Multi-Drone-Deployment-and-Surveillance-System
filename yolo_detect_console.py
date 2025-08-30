import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
import serial

# ------------------- LIDAR Setup -------------------
try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
except Exception as e:
    print(f"Warning: LiDAR not connected: {e}")
    ser = None

def read_lidar():
    """Read distance and strength from TF-Luna LiDAR"""
    if ser and ser.in_waiting >= 9:
        data = ser.read(9)
        if data[0] == 0x59 and data[1] == 0x59:  # Frame header
            dist = data[2] + (data[3] << 8)      # Distance in cm
            strength = data[4] + (data[5] << 8)  # Signal strength
            return dist, strength
    return None, None

# ------------------- Arguments -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='YOLO model path')
parser.add_argument('--source', required=True, help='Source (arducam0/usb0/video/file etc.)')
parser.add_argument('--resolution', default="640x480", help='WxH resolution')
args = parser.parse_args()

model = YOLO(args.model, task='detect')
labels = model.names
resW, resH = map(int, args.resolution.split('x'))

# ------------------- Source -------------------
if 'arducam' in args.source:
    idx = int(args.source[7:])
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
else:
    cap = cv2.VideoCapture(args.source)

# ------------------- Loop -------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Stream ended.")
        break

    results = model(frame, verbose=False)
    detections = results[0].boxes

    for det in detections:
        classidx = int(det.cls.item())
        classname = labels[classidx]
        conf = det.conf.item()

        if conf > 0.5 and classname.lower() == "drone":
            dist, strength = read_lidar()
            if dist is not None:
                print(f"Drone detected | Conf: {conf:.2f} | Distance: {dist} cm | Strength: {strength}")
            else:
                print(f"Drone detected | Conf: {conf:.2f} | LiDAR data not available")

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
