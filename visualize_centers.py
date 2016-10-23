# USAGE
# python visualize_centers.py --dataset ~/Desktop/ukbench_sample --features-db output/features.hdf5 \
#	--codebook output/vocab.cpickle --output output/vw_vis

# import the necessary packages
from __future__ import print_function
from pyimagesearch import ResultsMontage
from sklearn.metrics import pairwise
import numpy as np
import progressbar
import argparse
import _pickle as cPickle
import h5py
import cv2
import imutils

DEBUG = False

if DEBUG == False:
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
        ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
        ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
        ap.add_argument("-o", "--output", required=True, help="Path to output directory")
        args = vars(ap.parse_args())
        codebook=args["codebook"]
        features_db=args["features_db"]
        dataset=args["dataset"]
        output=args["output"]

else:
        codebook= "codebook\\vocab.cpickle"
        features_db= "features\\watchesFeatures.hdf5"
        dataset="datasets\\watches"
        output="output\\vw_vis"

# load the codebook and open the features database
vocab = cPickle.loads(open(codebook,"rb").read())
featuresDB = h5py.File(features_db, mode="r")
print("[INFO] starting distance computations...")

# initialize the visualizations dictionary and initialize the progress bar
vis = {i:[] for i in np.arange(0, len(vocab))}
widgets = ["Comparing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=featuresDB["image_ids"].shape[0], widgets=widgets).start()

# loop over the image IDs
for (i, imageID) in enumerate(featuresDB["image_ids"]):
        #print("Image nb{}: {}".format(i, imageID))
	# grab the rows for the features database and split them into keypoints and
	# feature vectors
	(start, end) = featuresDB["index"][i]
	rows = featuresDB["features"][start:end]
	(kps, descs) = (rows[:, :2], rows[:, 2:])

	# loop over each of the individual keypoints and feature vectors
	for (kp, features) in zip(kps, descs):
		# compute the distance between the feature vector and all clusters,
		# meaning that we'll have one distance value for each cluster
		D = pairwise.euclidean_distances(features.reshape(1, -1), Y=vocab)[0]

		# loop over the distances dictionary
		for j in np.arange(0, len(vocab)):
			# grab the set of top visualization results for the current
			# visual word and update the top reults with a tuple of the
			# distance, keypoint, and image ID
			topResults = vis.get(j)
			topResults.append((D[j], kp, imageID))

			# sort the top results list by their distance, keeping only
			# the best 16, then update the visualizations dictionary
			topResults = sorted(topResults, key=lambda r:r[0])[:16]
			vis[j] = topResults

	# update the progress bar
	pbar.update(i)

# close the features database
pbar.finish()
featuresDB.close()
print("[INFO] writing visualizations to file...")

# loop over the top results
for (vwID, results) in vis.items():
	# initialize the results montage
	montage = ResultsMontage((64, 64), 4, 16)

	# loop over the results
	for (_, (x, y), imageID) in results:
		# load the current image
		p = str(imageID)
		image = cv2.imread(p)
		image = imutils.resize(image, width=320)
		(h, w) = image.shape[:2]
		
		# extract a 8x8 region surrounding the keypoint
		(startX, endX) = (max(0, x - 16), min(w, x + 16))
		(startY, endY) = (max(0, y - 16), min(h, y + 16))
		roi = image[int(startY):int(endY), int(startX):int(endX)]
		
		# add the ROI to the montage
		montage.addResult(roi)

	# write the visualization to file
	p = "{}\\vis_{}.jpg".format(output, vwID)
	cv2.imwrite(p, cv2.cvtColor(montage.montage, cv2.COLOR_BGR2GRAY))
