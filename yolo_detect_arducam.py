import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
import serial

ser = None  # Will initialize later, after camera is confirmed working

def get_lidar_distance():
    global ser
    if ser is None:
        return None, None, None
    data = ser.read(9)
    if len(data) == 9 and data[0] == 0x59 and data[1] == 0x59:
        distance = data[2] + (data[3] << 8)   # in cm
        strength = data[4] + (data[5] << 8)
        temperature = (data[6] + (data[7] << 8)) / 8.0 - 256
        return distance, strength, temperature
    return None, None, None

# ----------------- Argument Parser -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), index of Picamera ("picamera0"), or Arducam ("arducam")', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")', default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), otherwise, match source resolution', default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.', action='store_true')
args = parser.parse_args()

# ----------------- Model Setup -----------------
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
elif 'arducam' in img_source:
    source_type = 'arducam'
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

if record:
    if source_type not in ['video','usb','arducam']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
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
    # Auto-detect the first available /dev/video device for Arducam
    arducam_idx = None
    for i in range(4):  # check /dev/video0 → /dev/video3
        test_cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if test_cap.isOpened():
            arducam_idx = i
            test_cap.release()
            break
    if arducam_idx is None:
        print('No Arducam device found.')
        sys.exit(0)
    print(f"Using Arducam at /dev/video{arducam_idx}")
    cap = cv2.VideoCapture(arducam_idx, cv2.CAP_V4L2)

    # Try to grab first frame to confirm camera works
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera failed at init")
        sys.exit(0)
    print("Camera OK, now opening LiDAR...")

    # Initialize LiDAR only after camera works
    ser = serial.Serial("/dev/serial0", 115200, timeout=1)

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# ----------------- Inference Loop -----------------
while True:
    t_start = time.perf_counter()

    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the USB camera. Exiting program.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if (frame is None):
            print('Unable to read frames from the Picamera. Exiting program.')
            break
    elif source_type == 'arducam':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the Arducam. Exiting program.')
            break

    if resize:
        frame = cv2.resize(frame,(resW,resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > float(min_thresh):
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            cv2.putText(frame, label, (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            object_count += 1

            # ------------- LiDAR Integration -------------
            if classname.lower() == "drone":
                dist, strength, temp = get_lidar_distance()
                if dist:
                    print(f"Drone detected at {dist} cm (Strength={strength}, Temp={temp:.1f}°C)")
                    cv2.putText(frame, f"Dist: {dist} cm", (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    if source_type in ['video', 'usb', 'picamera', 'arducam']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results',frame)
    if record: recorder.write(frame)

    if source_type in ['image','folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png',frame)

    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    avg_frame_rate = np.mean(frame_rate_buffer)

print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video','usb','arducam']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
