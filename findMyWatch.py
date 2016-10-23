import h5py
import cv2
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import argparse
from pathlib import Path
import imutils
from pyimagesearch.descriptors import RootSIFT
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.ir import BagOfVisualWords
import _pickle as cPickle

#this program takes an image as input and shows 10 similar watches
#It uses KMeans learning trained with a bovw hdf5 DB

DEBUG = False


#**********1.0 init some params**********
#----------------------------------------
featuresDB = h5py.File("features/watchesFeatures.hdf5")
bovwDB = h5py.File("bovw/watchesBovw.hdf5")
codebook="codebook/vocab.cpickle"
data=[]
IMAGE_WIDTH = 320
basePathWatches="datasets\\"

#load the histogram DB
for (i,imageID) in enumerate(bovwDB["bovw"]):
    hist = bovwDB["bovw"][i]
    data.append(hist)
print("[INFO] histograms of dataset loaded...")

#init the detector and descriptor (same as index_features.py)
detector = cv2.xfeatures2d.SURF_create()
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)

print("[INFO] detector initialized...")

# load the codebook vocabulary and initialize the bag-of-visual-words transformer (same as extract_bovw)
vocab = cPickle.loads(open(codebook, "rb").read())
bovw = BagOfVisualWords(vocab)
print("[INFO] bovw created...")

#parse the params
if DEBUG == False:
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image to find")
    ap.add_argument("-b", "--nb-bins", required=True, type= int, help="Number of categories of watches")
    args = vars(ap.parse_args())
    queryWatchFile = Path(args["image"])
    nbBins = args["nb_bins"]
else:
    queryWatchFile = Path("c:/Users/olivi/Google Drive/findmywatch/imagesWatches/olivierWatch3.jpg")
    nbBins = 500
    
 
# cluster the color histograms
clt = KMeans(n_clusters=nbBins)
labels = clt.fit_predict(data)

print("[INFO] clustering done...({} clusters)".format(nbBins))
 

#**********2.0 process the input image**********
#-----------------------------------------------

#load the image
queryWatch = cv2.imread(str(queryWatchFile))
print(str(queryWatchFile))


#normalize the image (same as in index_features)
queryWatch = imutils.resize(queryWatch, width=IMAGE_WIDTH)
cv2.imshow ("Queried Watch", queryWatch)
cv2.waitKey(3)
queryWatch = cv2.cvtColor(queryWatch, cv2.COLOR_BGR2GRAY)


#describe the image
(kps, descs) = dad.describe(queryWatch)
#classify the descriptors in an histogram
histQueryWatch = bovw.describe(descs)
#grab the label corresponding to this histogram
labelQueryWatch = clt.predict(histQueryWatch)

#*********3.0 show similar watches**********
#-------------------------------------------
#show all similar watches found in the catalog
#grab all image paths that are assigned to the query watch label
labelPathsIDs = np.where(labels == labelQueryWatch)

# loop over the image paths that belong to the current label
for ID in labelPathsIDs[0]:
        # load the image and display it
        imageFile=basePathWatches+str(featuresDB["image_ids"][ID])
        image = cv2.imread(imageFile)
        cv2.imshow("Cluster{}, Image #{}".format(labelQueryWatch, ID + 1), image)
        cv2.waitKey(0)
        # wait for a keypress and then close all open windows
cv2.destroyAllWindows()
