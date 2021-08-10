"""
    Drawing utils for object detection
"""

# Imports
import numpy as np
import cv2


def DrawBBoxesYOLO(image, idxs, boxes, classIDs):
    """ Draw Bounding Boxes """

    # Global Paths
    LABELS_PATH = "yolo/coco.names"

    # Labels
    LABELS = open(LABELS_PATH).read().strip().split("\n")

    # Colors for bounding boxes
    np.random.seed(7)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Filtered Bounding boxes
    rects = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            # (x, y) coordinates of the bounding box
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            box = (xmin, ymin, xmax, ymax) = np.array(
                (x, y, x + w, y + h)).astype("int")

            # If object is car
            if LABELS[classIDs[i]] == "car" or LABELS[classIDs[i]] == "truck":
                # add coordinates to filtered boxes
                rects.append(box)

                # Draw bounding box
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    return image, rects


def DrawBBoxesTF(image, boxes, idxs):
    rects = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
            rects.append(box)
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(
                box[2]), int(box[3])), (255, 0, 0), 3)

    return image, rects


def DrawBBoxes(image, boxes):
    if len(boxes) > 0:
        for box in boxes:
                xmin, ymin, xmax, ymax = box #list(map(int, box))
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
    
    return image

def DrawBBoxes_Text(image, boxes ,texts):
    if len(boxes) > 0:
        for box ,text in zip(boxes,texts):
                xmin, ymin, xmax, ymax = box #list(map(int, box))
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                cv2.putText(
                    image,
                    text,
                    (xmin - 10, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
    
    return image