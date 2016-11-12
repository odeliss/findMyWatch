# USAGE
# python index_features.py --dataset ~/Desktop/ukbench_sample --features-db output/features.hdf5

# import the necessary packages
#[12-11-2016] added scraped data in hdf5 database

from __future__ import print_function
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.descriptors import RootSIFT
from pyimagesearch.indexer import FeatureIndexer
from imutils import paths
import argparse
import imutils
import cv2
from pathlib import Path
import csv

DEBUG = True
IMAGE_WIDTH = 320
#TO DO: put the constant in a common constant file

if DEBUG == False:
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", required=True,
                help="Path to the directory that contains the images to be indexed")
        ap.add_argument("-f", "--features-db", required=True,
                help="Path to where the features database will be stored")
        ap.add_argument("-a", "--approx-images", type=int, default=500,
                help="Approximate # of images in the dataset")
        ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,
                help="Maximum buffer size for # of features to be stored in memory")
        args = vars(ap.parse_args())
        
        dataset = args["dataset"]
        featuresDb = args["features_db"]
        approxImages = args["approx_images"]
        maxBufferSize = args["max_buffer_size"]
else:
        #dataset = "UKBenchDataset/ukbench_quiz"
        dataset = "datasets/watches/"
        featuresDb = "features/featuresquiz.hdf5"
        approxImages = 1000
        maxBufferSize = 800000
# initialize the keypoint detector, local invariant descriptor, and the descriptor
'''Adapted for CV3 compliance
detector = cv2.FeatureDetector_create("SURF")
'''
detector = cv2.xfeatures2d.SURF_create()
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)

#load the CSV file with additional data scraped into a dictionary with key = filename
scrapedDataFile = open(str(dataset)+"_images.csv")
scrapedDataReader = csv.reader(scrapedDataFile)
scrapedData= list(scrapedDataReader)
scrapedDataDict={}

#[12-11-2016]
for row in scrapedData: #loop on the CSV file rows and create a dictionnary of additional scraped data
        index=row[0].rfind("\\")+1
        scrapedDataDict[row[0][index+1:]]=row[1]

# initialize the feature indexer
fi = FeatureIndexer(featuresDb, estNumImages=approxImages,
	maxBufferSize=maxBufferSize, verbose=True)

# loop over the images in the dataset
for (i, imagePath) in enumerate(paths.list_images(dataset)):
        # check to see if progress should be displayed
        if i > 0 and i % 10 == 0:
                fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

        # extract the image filename (i.e. the unique image ID) from the image
        # path, then load the image itself
        filename = imagePath[imagePath.rfind("\\") + 1:]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=IMAGE_WIDTH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # describe the image
        (kps, descs) = dad.describe(image)

        # if either the keypoints or descriptors are None, then ignore the image
        if kps is None or descs is None:
                continue

        # index the features
        #[12-11-2016]
        indexDict = filename.rfind("/")+2 #+2 coz some_ there...
        fi.add(filename, kps, descs, scrapedDataDict[filename[indexDict:]])
# finish the indexing process
fi.finish()
