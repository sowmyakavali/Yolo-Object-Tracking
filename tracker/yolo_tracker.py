"""
    Object Tracker
"""
# Update sys env variable
import os
# import re
import sys

base_path = os.path.realpath("..")
sys.path.append(base_path)

# Imports
import os
import datetime
import ntpath
import argparse
import cv2
import dlib
import pandas as pd
# import torch
# import numpy

id_alias = {"signboard": "SB", "streetlight": "ST", "plantation": "PL", "bush": "Bush"}

dis_rate = {"signboard": 20, "streetlight": 5, "plantation": 5, "bush": 5}

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", required=True, help="Class of the object to track")
ap.add_argument("-v", "--video", required=True, help="path to input image")
ap.add_argument(
    "-f", "--frame", type=int, default=8, help="nth frame to run inference on"
)
ap.add_argument(
    "-r", "--resize", type=int, default=1, help="Resize on a factor of value"
)
ap.add_argument(
    "-p",
    "--confidence",
    type=float,
    default=0.75,
    help="minimum probability to filter weak detections default(0.9)",
)
ap.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.5,
    help="threshold for non max suppression",
)
ap.add_argument(
    "-dm", "--depth", default="false", help="Distance between object and camera(apprx.)"
)
ap.add_argument(
    "--disappeared",
    type=int,
    default=None,
    help="max disappeared number before removing live objects",
)
ap.add_argument(
    "--model",  default='yolov5', help="Tensorflow version - accepted values 1, 2"
)
ap.add_argument("--roi", type=int, default=2, help="Reduced roi threshold")
ap.add_argument(
    "-d", "--display", default="true", help="Display the infered images & frames"
)
ap.add_argument("-s", "--save", default="false", help="save the output")
ap.add_argument("-ch", "--chainage", default="false", help="chainage data file")
args = vars(ap.parse_args())

# Set object class label & base path
os.environ["BASE_PATH"] = base_path
os.environ["OBJECT_CLASS_LABEL"] = args["class"]

# Primary data load
latlong = pd.read_csv(args["chainage"])
latlong.index = latlong["frame_index"]
del latlong["frame_index"]

print('[INFO] Your video : {}'.format(args["video"][3:]))

# Additional imports
from centroid_tracker import CentroidTracker
from lib.utils import DrawBBoxesTF, DrawBBoxes, DrawBBoxes_Text
from lib.bbox_filters import MaxBBox, ReducedROI, NMSIdxToBBox
from lib.mock_chainage import ChainageSequence, setInterval
from lib.depth_estimator import NearestObject, DrawDepthMeter
from utils.datasets import LoadStreams, LoadImages

# yolo version
if args["model"] == 'yolov5' :    
    from detector.yolov5_object_detector import Inference
else :    
    from detector.yolo_object_detector import Inference



def init():

    # Set default disappearence rate
    if args["disappeared"] == None:
        args["disappeared"] = int(dis_rate[args["class"]])

    # Initialize centroid tracker
    ct = CentroidTracker(args["disappeared"])

    # Initialise a list to store all trackers
    trackers = []

    (W, H) = (None, None)
    writer = None

    frameCount = -1
    totalElapsed = 0

    # Start inference on every nth frame
    print("[INFO] Looking for {}(s)".format(args["class"]))

    try:
        # Read frame
        stride = int(os.environ['STRIDE'])
        dataset = LoadImages(args["video"], img_size=640, stride=stride)
        for path, frame, im0s, vid_cap in dataset:
            cv2.waitKey(1000) 

            # Update frame count
            frameCount += 1

            rects = []
            
            # Store frame dimensions
            if W is None or H is None:
                (H, W) = im0s.shape[:2]
                H = int(H / args["resize"])
                W = int(W / args["resize"])


            # # Create region to detet objects (For bushes the right region is right bottom part of image)
            # cv2.line(im0s, (W//2, H // 3), (W, H // 3), (0, 255, 255), 2)
            # cv2.line(im0s, (W//2, H//3), (W//2, H), (0, 255, 255), 2)

            # Get corresponding latitudes and longitudes of frame
            latlong_frame = latlong.loc[frameCount]
            lat_frame = round(float(latlong_frame.latitude), 7)
            long_frame = round(float(latlong_frame.longitude), 7)
            chainage = float(latlong_frame.chainage)

            # Generate a text to display on the frame
            latlong_text = str(lat_frame) + " " + str(long_frame) + " " + str(chainage)

            # Display generated latlong text
            cv2.putText(
                im0s,
                latlong_text,
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Get nth frame
            if frameCount % args["frame"] == 0 or frameCount == 0:

                # Update global tracker every `n` frame
                trackers = []

                # Inference
                boxes, confidences, elapsed = Inference(frame, im0s, path, dataset)

                # Update total elapsed time
                totalElapsed += elapsed
                
                # Go through each box
                for rect in boxes:

                    # Create a correlation tracker instance for each object
                    tracker = dlib.correlation_tracker()

                    xmin, ymin, xmax, ymax = rect
                    rect = dlib.rectangle(xmin, ymin, xmax, ymax)

                    # Start the dlib correlation tracker
                    tracker.start_track(im0s, rect)

                    # Append tracker to list of existing trackers
                    trackers.append(tracker)

                # Depth Estimator
                if args["depth"] != "false":
                    # get nearest object
                    nearest_box = NearestObject(rects, H, W)
                    if nearest_box != None:
                        DrawDepthMeter(frame, nearest_box)
            else:
                for tracker in trackers:
                    # update the tracker and grab the updated position
                    tracker.update(im0s)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    rects.append((startX, startY, endX, endY))

            # Update centroid tracker with bounding boxes
            objects = ct.update(
                rects, frameCount, chainage, [lat_frame, long_frame]
            )
            # else:
            # print("[INFO] No objects found at frame {}".format(frameCount))
            # None

            # Draw final filtered detections
            DrawBBoxes(im0s, rects)

            # Loop over tracked objects
            for (objectID, centroid) in objects.items():
                # Write the ID of the object to output frame
                text = "{}_{}".format(id_alias[args["class"]], objectID)
                cv2.putText(
                    im0s,
                    text,
                    (centroid[0] - 10, centroid[1] - 10),#(xmin - 10, ymin - 10), #
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 200, 0),
                    2,
                )

                # Draw centroid point on the output frame
                cv2.circle(im0s, (centroid[0], centroid[1]), 2, (255, 255, 0), -1)

            if args["display"] == "true":
                # Display Frame
                cv2.imshow("Image", im0s)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args["save"] != "false":
                print(args["video"].split(".")[0])
                # Initialize video writer
                if writer is None:
                    Vid_savePath = os.path.join(
                        "..",
                        "results",
                        args["class"]
                        + "_"
                        + args["video"][3:].split(".")[0]
                        + "_"
                        + str(datetime.datetime.now()).split(" ")[0]
                        + ".mp4",
                    )
                    print("[INFO] Writing frames at", Vid_savePath)

                    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                    writer = cv2.VideoWriter(Vid_savePath, fourcc, 20, (W, H), True)

                # write output frame
                print('Writing frame to video')
                writer.write(im0s)

    except KeyboardInterrupt:
        print("")
        print("[EXCP] Keyboard interrupt")
        print("[INFO] Cleanup initialized")

    print(
        "[INFO] Infered {} frames in {:.2f} secs, FPS: {:.2f}".format(
            frameCount, totalElapsed, frameCount / totalElapsed
        )
    )

    savePath = os.path.join(
        "..",
        "results",
        args["class"] 
        + "_" 
        + args["video"][3:].split(".")[0]  +
        ntpath.basename(
            "_" + str(datetime.datetime.now()).split(" ")[0] 
             + ".csv"
        ),
    )
    ct.objectsData["id"] = (
        id_alias[args["class"]] + "_" + ct.objectsData["id"].astype("str")
    )
    ct.objectsData.to_csv(savePath, index=False)
    print("[INFO] Location map stored at {}".format(savePath))

    # Clean up
    cv2.destroyAllWindows()

    if writer != None:
        writer.release()
    print("[INFO] video stored at {}".format(Vid_savePath))    


if __name__ == "__main__":
    # If not imported start the tracker
    init()
