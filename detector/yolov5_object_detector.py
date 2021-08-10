import os
import sys
import time
import torch
import numpy as np

# sys.path.append(os.path.realpath("../lib/yolo"))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size,  non_max_suppression, scale_coords, xyxy2xywh,  set_logging
from utils.torch_utils import select_device


# Important arguments
device, imgsz, augment = 'cpu', 640, 'store_true'
weights, assettype, model, stride,  = None, None, None, None


def init_model(label, base_path):

    global weights, assettype, model, stride, imgsz, device
    assettype = label
    weights = os.path.join(base_path, label, 'best.pt')
    # Initialize
    set_logging()
    device = select_device(device) # you can give 0 or 1 if you have gpu  
    # load FP32 model
    model = attempt_load(weights, map_location=device)  
    # model stride
    stride = int(model.stride.max())  
    # check img_size
    imgsz = check_img_size(640, s=stride) 


def Inference(im0s, Height, Width ):

        # Padded resize
        img = letterbox(im0s, imgsz, stride=stride)[0]
        # Convert  BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)    

        # Image to numpy array
        img = torch.from_numpy(img).to(device)
        # uint8 to fp16/32
        img =  img.float()  
        # 0 - 255 to 0.0 - 1.0
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t0 = time.time()
        pred = model(img, augment=augment)[0]     
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.5)
        elapsed = time.time() - t0

        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        # detections per image
        for i, det in enumerate(pred):  
            # normalization gain whwh
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0s size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

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

                    box = [xmin, ymin, xmax, ymax]
                    boxes.append(box)
                    confidences.append(float(conf))
                    class_ids.append(int(cls))                              

        return boxes ,confidences ,elapsed 
