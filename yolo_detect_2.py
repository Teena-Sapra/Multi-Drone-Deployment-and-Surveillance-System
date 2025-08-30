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
# Initialize TF-Luna LiDAR on serial0
try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
except Exception as e:
    print(f"Warning: LiDAR not connected or failed to open serial port: {e}")
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

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")')
parser.add_argument('--source', required=True, help='Image source (image/video/usbX/picameraX/arducamX)')
parser.add_argument('--thresh', default=0.5, help='Minimum confidence threshold')
parser.add_argument('--resolution', default=None, help='Resolution WxH (example: "640x480")')
parser.add_argument('--record', action='store_true', help='Record results as "demo1.avi" (requires --resolution)')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model path invalid or not found.')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# ------------------- Source Setup -------------------
img_ext_list = ['.jpg','.jpeg','.png','.bmp','.JPG','.JPEG','.PNG','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list: source_type = 'image'
    elif ext in vid_ext_list: source_type = 'video'
    else:
        print(f'Unsupported file extension {ext}.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
elif 'arducam' in img_source:
    source_type = 'arducam'
    arducam_idx = int(img_source[7:])
else:
    print(f'Invalid input source: {img_source}')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Recording setup
if record:
    if source_type not in ['video','usb','arducam']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Capture setup
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list: imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()
elif source_type == 'arducam':
    cap = cv2.VideoCapture(arducam_idx, cv2.CAP_V4L2)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# ------------------- Detection Loop -------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate, frame_rate_buffer, fps_avg_len, img_count = 0, [], 200, 0

while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type == 'video' or source_type == 'usb' or source_type == 'arducam':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Stream ended or failed. Exiting.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Unable to read Picamera. Exiting.')
            break

    if resize:
        frame = cv2.resize(frame,(resW,resH))

    # YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'

            # If drone detected, get LiDAR reading
            if classname.lower() == "drone":
                dist, strength = read_lidar()
                if dist is not None:
                    label += f' | {dist}cm | Str:{strength}'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),
                          (xmin+labelSize[0], label_ymin+baseLine-10),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # Overlay FPS + object count
    if source_type in ['video', 'usb', 'picamera', 'arducam']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('YOLO + LiDAR Detection', frame)
    if record: recorder.write(frame)

    key = cv2.waitKey(0 if source_type in ['image','folder'] else 5)
    if key in [ord('q'), ord('Q')]: break
    elif key in [ord('s'), ord('S')]: cv2.waitKey()
    elif key in [ord('p'), ord('P')]: cv2.imwrite('capture.png',frame)

    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len: frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# ------------------- Cleanup -------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video','usb','arducam']: cap.release()
elif source_type == 'picamera': cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
