# USAGE
# python index_features.py --dataset ~/Desktop/ukbench_sample --features-db output/features.hdf5

# import the necessary packages
#[12-11-2016] added scraped data in hdf5 database
#[28-11-2016] added title as new scraped data
#[26-12-2016] added append switch to append data in existing DB
#TO DO: put the constant in a common constant file

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

DEBUG = False
IMAGE_WIDTH = 320
ROW_REFERENCE = 1 #location in CSV line of the reference of the watch
ROW_TITLE = 2 #location in the CSV line of the title of the watch

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
        #[26-12-2016 +++]
        ap.add_argument("-ap", "--appendToFile", type=bool, 
                help="True indicates that data must be added to existing file")
        #[26-12-2016 ---]
        args = vars(ap.parse_args())
        
        dataset = args["dataset"]
        featuresDb = args["features_db"]
        approxImages = args["approx_images"]
        maxBufferSize = args["max_buffer_size"]
        #[26-12-2016 +++]
        appendToFile = args["appendToFile"]
        #[26-12-2016 ---]
else:
        #dataset = "UKBenchDataset/ukbench_quiz"
        dataset = "testdata/datasets/watches/"
        featuresDb = "testdata/features/watchesFeatures.hdf5"
        approxImages = 2
        maxBufferSize = 80000
        #[26-12-2016 +++]
        appendToFile = False
        #[26-12-2016 ---]
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
        scrapedDataItem =[]
        index=row[0].rfind("\\")+1 #the dictionary key is named as per the filename
        #the value are stored in a str with items separated by #.
        # Not found a better way to store in hdf5
        scrapedDataDict[row[0][index+1:]]=str(row[ROW_REFERENCE])+"#"+str(row[ROW_TITLE]) #store the reference into item 1 of the dictionary with key =filename
        

# initialize the feature indexer
#[26-12-2016 +++]
fi = FeatureIndexer(featuresDb, appendToFile, estNumImages=approxImages,
	maxBufferSize=maxBufferSize, verbose=True)
#[26-12-2016 ---]

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
