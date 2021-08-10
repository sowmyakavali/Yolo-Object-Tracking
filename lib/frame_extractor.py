"""
    Extract Frames for Annotation
"""

# Imports
import os
import datetime
import ntpath
import argparse
import cv2

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                required=True,
                help="path to input image")
ap.add_argument("-f", "--frame",
                type=int,
                default=8,
                help="nth frame to run inference on")
ap.add_argument("-r", "--resize",
                type=int,
                default=1,
                help="Resize on a factor of value")
ap.add_argument("-o", "--output",
                default="./data",
                help="output folder where data is saved")
args = vars(ap.parse_args())

def extract():
    # Initialize video stream
    vs = cv2.VideoCapture(args["video"])
    vs = vs.set(cv2.CAP_PROP_EXPOSURE, 40)
    fps = vs.get(cv2.CAP_PROP_FPS)
    nth_frame = int(fps/2)
    print('FPS {} '.format(fps))
    (W, H) = (None, None)
    writer = None

    frameCount = 0
    totalFrames = 0

    print("[INFO] Writing frames at {}".format(os.path.abspath(args["output"])))

    try:
        while True:
            # Raed frame
            (grabbed, frame) = vs.read()

            # Break if grabbed is false
            if not grabbed:
                break

            # Update frame count
            frameCount += 1

            # Store frame dimensions
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                H = int(H / args["resize"])
                W = int(W / args["resize"])

            # Resize frame
            frame = cv2.resize(frame, (W, H))
            
            if frameCount % nth_frame == 0:
                # Update total frames
                totalFrames += 1

                # Save Frame
                path = args["video"][3:].split(".")[0]
                img_name = path+'_'+str(frameCount) + ".jpg"
                img_path = os.path.join(args["output"], img_name)
                cv2.imwrite(img_path, frame)

    except KeyboardInterrupt:
        print("")
        print("[EXCP] Keyboard interrupt")
        print("[INFO] Cleanup initialized")

    print("[INFO] Extracted {} frames".format(
        totalFrames))
    
    # Clean up
    vs.release()
    if writer != None:
        writer.release()

if __name__ == "__main__":
    extract()
