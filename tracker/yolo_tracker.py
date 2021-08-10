"""
    Object Tracker
"""

# Imports
import os
import cv2
import sys

sys.path.append(os.path.realpath(".."))

from lib.utils import DrawBBoxes
from lib.bbox_filters import NMSIdxToBBox
from tracker.centroid_tracker import CentroidTracker
from lib.depth_estimator import NearestObject, DrawDepthMeter

# Golbal variables
id_alias = {"signboard": "SB", "streetlight": "ST", "plantation": "PL"}
dis_rate = {"signboard": 20, "streetlight": 8, "plantation": 5}
confs = {"signboard": 0.5, "streetlight": 0.7, "plantation": 0.5}

base_path = os.path.realpath("..")

from detector.yolov5_object_detector import Inference, init_model


def init(vid_path,
         label,
         resize_ratio=1,
         confidence=None,
         threshold=0.5,
         depth=False,
         disappear_rate=None,
         view_image=True,
         save_video=True):

    # Initialize detector
    init_model(label, base_path)

    # Initialize video stream
    vs = cv2.VideoCapture(vid_path)

    # Set default disappearence rate
    if disappear_rate == None:
        # vs_fps = vs.get(cv2.CAP_PROP_FPS)
        disappear_rate = int(dis_rate[label])

    # Set default confidence
    if confidence == None:
        confidence = float(confs[label])

    # Initialize centroid tracker
    ct = CentroidTracker(disappear_rate)

    (W, H) = (None, None)
    writer = None

    frameCount = -1
    totalElapsed = 0

    # Start inference on every nth frame
    print("[INFO] Looking for {}(s)".format(label))
    try:
        while True:
            # Read frame
            (grabbed, frame) = vs.read()

            # Break if grabbed is false
            if not grabbed:
                break

            # Update frame count
            frameCount += 1

            rects = []

            # Store frame dimensions
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                H = int(H / resize_ratio)
                W = int(W / resize_ratio)

            frame = cv2.resize(frame, (W, H))

            # Inference
            boxes, confidences, elapsed = Inference(frame, H, W)

            # Update total elapsed time
            totalElapsed += elapsed

            # Non Max Suppression
            idxs = cv2.dnn.NMSBoxes(
                boxes, confidences, confidence, threshold
            )

            # Extracting boxes from NMS idxs
            rects = NMSIdxToBBox(boxes, idxs)

            # Depth Estimator
            if depth != False:
                # get nearest object
                nearest_box = NearestObject(rects, H, W)
                if nearest_box != None:
                    DrawDepthMeter(frame, nearest_box)

            # Update centroid tracker with bounding boxes
            objects = ct.update(rects, frameCount)

            if view_image or save_video:
                # Draw final filtered detections
                DrawBBoxes(frame, rects)

                for (objectID, centroid) in objects.items():
                    # Write the ID of the object to output frame
                    text = "{}_{}".format(id_alias[label], objectID)
                    cv2.putText(
                        frame,
                        text,
                        (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )

                    # Draw centroid point on the output frame
                    cv2.circle(
                        frame, (centroid[0], centroid[1]), 2, (255, 255, 0), -1)

            if view_image:
                # Display Frame
                cv2.imshow("Image", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if save_video:
                # Initialize video writer
                if writer is None:
                    savePath = vid_path.replace('.mp4', '_' + label + 's.mp4')

                    print("[INFO] Writing frames at", savePath)

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        savePath, fourcc, 10, (W, H), True)

                # write output frame
                writer.write(frame)

    except KeyboardInterrupt:
        print("[EXCP] Keyboard interrupt")
        print("[INFO] Cleanup initialized")

    # Clean up
    cv2.destroyAllWindows()
    vs.release()
    if writer != None:
        writer.release()

    print(
        "[INFO] Infered {} frames in {:.2f} secs, FPS: {:.2f}".format(
            frameCount, totalElapsed, frameCount / totalElapsed
        )
    )

    # Return unique chainages
    # print(list(ct.objectsData["chainage"].unique()))
    # return list(ct.objectsData["chainage"].unique())


"""
Testing instructions

cd video-upload

from tracker.object_tracker import init
chainages = init(vid_path=r'D:\assets\data\jul16-14-27\video.mp4',label='signboard', chainage_data=r'D:\assets\data\jul16-14-27\video.csv')
"""
init(vid_path=r'..\video.mp4',label='signboard')
