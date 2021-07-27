"""Centroid Tracker
"""

# Imports
from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd

from lib.mock_chainage import ChainageSequence, setInterval


class CentroidTracker:
    """Centroid  Traceker Class"""

    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0

        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # Data Frame to store objects first and last frame along with object id
        self.columns = [
            "id",
            "latitude",
            "longitude",
            "chainage",
            "first_frame",
            "f_lat",
            "f_long",
            "f_chainage",
            "last_frame",
            "l_lat",
            "l_long",
            "l_chainage",
        ]
        self.objectsData = pd.DataFrame(columns=self.columns)
        # Data Frame to store chinage and corresponding object
        self.chainageData = pd.DataFrame(columns=["chainage", "id"])

        self.maxDisappeared = maxDisappeared

    def register(self, centroid, frameNo, chainage, lat_long):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0

        # Add a row to objectsData with first frame & first chainage
        df = pd.DataFrame(
            [
                [
                    self.nextObjectID,
                    np.nan,
                    np.nan,
                    np.nan,
                    frameNo,
                    lat_long[0],
                    lat_long[1],
                    chainage,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ],
            columns=self.columns,
        )
        self.objectsData = self.objectsData.append(df)

        self.nextObjectID += 1

    def deregister(self, objectID, frameNo, chainage, lat_long):
        # Update the last frame of the row corresponding to objectid
        
        self.objectsData.loc[self.objectsData.id == objectID, "latitude"] = lat_long[0]
        self.objectsData.loc[self.objectsData.id == objectID, "longitude"] = lat_long[1]
        self.objectsData.loc[self.objectsData.id == objectID, "chainage"] = chainage

        self.objectsData.loc[self.objectsData.id == objectID, "last_frame"] = frameNo
        self.objectsData.loc[self.objectsData.id == objectID, "l_lat"] = lat_long[0]
        self.objectsData.loc[self.objectsData.id == objectID, "l_long"] = lat_long[1]
        self.objectsData.loc[self.objectsData.id == objectID, "l_chainage"] = chainage

        # # Update the last chainage of the row corresponding to objectid
        # self.objectsData.loc[self.objectsData.id ==
        #                      objectID, 'end_chainage'] = chainage

        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects, frameNo, chainage, lat_long):
        """
        Parameters:
        rects: [(startX, startY, endX, endY), ...]

        Returns:
        objects
        """
        # No bounding boxes found
        if len(rects) == 0:
            # Increment disappeared values of existing tracked objects
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # Deregister if maxDisappeared value is reached
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID, frameNo, chainage, lat_long)

            # return the set of trackable objects
            return self.objects

        # Initialize centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # Update centroids
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # Register objects
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], frameNo, chainage, lat_long)

        # Update objects centroids
        else:
            # Get objects and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute euclidean distances between all pairs of objectCentroids  and inputCentroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Find the smallest value in each row,
            # Sort the row indexes based on their minimum values so that the row with the smallest value is at the *front* of the index list
            rows = D.min(axis=1).argsort()

            # Find the smallest value in each column,
            # Sort using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # Keep track of already examined rows and column indexes
            usedRows = set()
            usedCols = set()

            # Loop over (row, column)
            for (row, col) in zip(rows, cols):
                # Ignore already examined rows or columns
                if row in usedRows or col in usedCols:
                    continue

                # Grab the object ID for the current row,
                # Set its new centroid,
                # Reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # update usedRows & usedCols
                usedRows.add(row)
                usedCols.add(col)

            # Get un examined row and column index
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # When object centroids is equal or greater than input centroids
            # Check if some of these objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # Loop over the unused row indexes
                for row in unusedRows:
                    # Grab the object ID for the corresponding row index
                    # Increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # Deregister if maxDisappeared threshold is reached
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID, frameNo, chainage, lat_long)
            # When input centroids are greater than existing object centroids
            # Register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], frameNo, chainage, lat_long)

        # return the set of trackable objects
        return self.objects
