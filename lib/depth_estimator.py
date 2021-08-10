import math
import cv2


def NearestObject(boxes, H, W):
    n_box = None

    max_height = 0

    for box in boxes:
        x_min, y_min, x_max, y_max = list(map(int, box))
        height = y_max - y_min
        if height > max_height:
            max_height = height
            n_box = box

    return n_box


def CalculateDepthMeter(n_box, img=None, img_height=None, box_height=None):
    if img != None:
        img_height, img_width = img.shape[:2]

        x_min, y_min, x_max, y_max = n_box
        box_height = y_max - y_min

    depth_meter = math.pow(img_height - box_height, 2) / img_height / 100

    return depth_meter


def DrawDepthMeter(img, n_box):
    img_height, img_width = img.shape[:2]

    x_min, y_min, x_max, y_max = n_box
    box_height = y_max - y_min

    depth_meter = CalculateDepthMeter(
        n_box, img_height=img_height, box_height=box_height
    )

    ground_centroid = (img_width // 2, img_height)
    box_centroid = (int((x_max + x_min) // 2), int((y_max + y_min) // 2))

    text = "{} dm".format(round(depth_meter, 2))
    x_text = box_centroid[0] + ground_centroid[0]
    y_text = box_centroid[1] + ground_centroid[1]
    text_padding = 10
    text_centroid = (x_text // 2 + text_padding, y_text // 2)

    img = cv2.line(img, ground_centroid, box_centroid, (255, 0, 0), 2)
    img = cv2.putText(
        img,
        text,
        text_centroid,
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    return img