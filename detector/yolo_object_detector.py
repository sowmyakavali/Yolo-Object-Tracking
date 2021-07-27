"""
    Object Detector
"""

# Imports
import time
import numpy as np
import cv2

from centroid_tracker import CentroidTracker

# Global Paths
CONFIG_PATH = "yolo/yolov3.cfg"
WEIGHTS_PATH = "yolo/yolov3.weights"

# Parameters
CONFIDENCE = 0.5

# Load yolo weights
print("[INFO] Loading YOLO")
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

# Determine output layers of YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def Inference(blob, H, W):
    """ Inference """
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    elapsed = time.time() - start

    # Bounding boxes & Confidences
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            # Class & Confidence associated with the detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter detections with confidence threshold
            if confidence > CONFIDENCE:
             # (x, y) coordinates of the bounding box
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Append boxes, cofindence and classes
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    return [boxes, confidences, classIDs, elapsed]
