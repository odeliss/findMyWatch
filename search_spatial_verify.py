# import the necessary packages
from __future__ import print_function
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.descriptors import RootSIFT
from pyimagesearch.ir import BagOfVisualWords
from pyimagesearch.ir import SpatialVerifier
from pyimagesearch.ir import Searcher
from pyimagesearch import ResultsMontage
from scipy.spatial import distance
from redis import Redis
import argparse
import cPickle
import imutils
import json
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to the bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-i", "--idf", required=True, help="Path to inverted document frequencies array")
ap.add_argument("-r", "--relevant", required=True, help = "Path to relevant dictionary")
ap.add_argument("-q", "--query", required=True, help="Path to the query image")
args = vars(ap.parse_args())
