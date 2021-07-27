# import argparse
import time
# from pathlib import Path

import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, source
# import numpy

from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# Get object label from environment variable
label = os.environ['OBJECT_CLASS_LABEL']
base_path = os.environ['BASE_PATH']

weights = os.path.join(base_path,label,'best.pt')
device = 'cpu'
imgsz = 640
augment = None
conf_thres = 0.55
iou_thres = 0.50


# Initialize
set_logging()
device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
    model.half()  # to FP16

os.environ["STRIDE"] = str(stride)

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

# If you want to display detections
view_img = False

def Inference(img ,im0s, path ,dataset ):

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Image to numpy array
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t0 = time.time()
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t2 = time_synchronized()

        elapsed = time.time() - t0

        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Get Image dimensions
                Height = im0.shape[0]
                Width = im0.shape[1]

                # Write results
                for *xyxy, conf, cls in reversed(det):  

                    # convert coordinates from tensor to list # normalized xywh
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    x, y, w, h =xywh[0] ,xywh[1] ,xywh[2] ,xywh[3]

                    # convert boxes from normalized to normal (pascal format)
                    xmin = int((x*Width) - (w*Width)/2.0)
                    ymin = int((y*Height) - (h*Height)/2.0)
                    xmax = int((x*Width) + (w*Width)/2.0)
                    ymax = int((y*Height) + (h*Height)/2.0)

                    if label == 'bush' :
                        # Append boxes which are at right bottom only (it is for bushes)
                        if xmin >= Width//3 and ymin >= Height//3 :
                            if xmax <= Width and ymax <= Height :
                                box = [xmin, ymin, xmax, ymax]     
                                # Reference view
                                if view_img:
                                    cv2.rectangle(im0, (xmin,ymin),(xmax,ymax), (255, 255, 0), 2)

                                # Store bboxes
                                boxes.append(box)
                                confidences.append(float(conf))
                                class_ids.append(int(cls))                                                            
                    else :
                        box = [xmin, ymin, xmax, ymax]

                        # Reference view
                        if view_img:
                            cv2.rectangle(im0, (xmin,ymin),(xmax,ymax), (255, 255, 0), 2)

                        # Store bboxes
                        boxes.append(box)
                        confidences.append(float(conf))
                        class_ids.append(int(cls))  

            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond                                 

        return boxes ,confidences ,elapsed                  