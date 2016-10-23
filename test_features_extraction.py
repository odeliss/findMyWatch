# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import h5py
import imutils

features_DB = "features\\watchesFeatures.hdf5"

for i in range (0,1000):
    #extract 100th image data from the hdf5 file
    db = h5py.File(features_DB, mode="r")
    imageID=db["image_ids"][i]
    (start,end)=db["index"][i]

    # load the image and convert it to grayscale
    image = cv2.imread(imageID)
    image = imutils.resize(image, width=320)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # Get the Keypoints of the image
    rows=db["features"][start:end]
    kps = rows[:,:2]
    #print("# of keypoints: {}".format(len(kps)))

     
    # loop over the keypoints and draw them
    for kp in kps:
            r = int(0.5 * kp.size)
            (x, y) = np.int0(kp)
            cv2.circle(image, (x, y), r, (0, 255, 255), 2)
     
    # show the image
    cv2.imshow("Images", np.hstack([orig, image]))
    cv2.waitKey(0)
