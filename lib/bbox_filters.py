"""
Extract Max Bounding Box from current 
"""


def MaxBBox(boxes):
    max_height = 0
    max_box = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        height = ymax - ymin
        if height > max_height:
            max_height = height
            max_box = box
    return max_box


def ReducedROI(boxes, H, W, roi=2):
    rects = []
    for box in boxes:
        if box[0] > (W / roi):
            rects.append(box)

    return rects


def NMSIdxToBBox(boxes, idxs):
    rects = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
            rects.append(box)

    return rects