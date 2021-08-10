"""
    Object Tracker
"""

# Imports
import os
import cv2
import sys
import argparse

base_path = os.path.realpath("..")
sys.path.append(base_path)

from lib.utils import DrawBBoxes
from lib.bbox_filters import NMSIdxToBBox
from tracker.centroid_tracker import CentroidTracker
from lib.depth_estimator import NearestObject, DrawDepthMeter

# Import detector
from detector.yolov5_object_detector import Inference, init_model


def init_tracker(vid_path,
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
        disappear_rate = 10 #int(dis_rate[label])

    # Set default confidence
    if confidence == None:
        confidence = 0.5

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
                    text = "{}_{}".format(label, objectID)
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

if __name__ =="__main__":
    ap = argparse.ArgumentParser()    
    ap.add_argument("-v", "--video",  help="input video to track")
    ap.add_argument("-l", "--label", default="false", help="give the label which you want to track")
    ap.add_argument("-dr", "--disapper_rate", default=10, help="Based on objects closeness give value, if object are very closer give less value")
    ap.add_argument("-c", "--confidence", default=0.5, help="minimum confidence of object should have to track")
    args = vars(ap.parse_args())
    init_tracker(vid_path=args["video"], label=args["label"], disapper_rate=args["disapper_rate"], confidence=args["confidence"])
