*CBIR watches matching. Give an image in input and you receive a list of similar watches

|--- PYIMAGESEARCH
|    |--- __init__.py
|    |--- descriptors
|    |    |---- __init__.py
|    |    |--- detectanddescribe.py
|    |    |--- rootsift.py
|    |--- indexer
|    |    |---- __init__.py
|    |    |--- baseindexer.py
|    |    |--- bovw_indexer.py
|    |    |--- featureindexer.py
|    |--- ir
|    |    |---- __init__.py
|    |    |--- bagofvisualwords.py
|    |    |--- vocabulary.py
|--- BOVW: reference histogram for the dataset
|--- CODEBOOK: visual words serving as reference (centroid of the features)
|--- WATCHES: Reference images
|    |--- __watch_dddddd.jpg
|    |...
|--- FEATURES: the features and keypoints for all images are stored there
|    |--- watchesFeatures.hdf5
|--- cluster_features.py
|--- extract_bovw.py
|--- index_features.py
|--- visualize_centers.py


STEP_1: store keypoints and features for all the watches in the dataset WATCHES
===============================================================================		
(1) run index_features.py with following parameters
---------------------------------------------------
	--dataset: directory WATCHES containing the images
	--features_db: directory FEATURES and filename of the DB that will contain the image_ID, start and end position of the features, list fo features
	--approx_images: apporximate number of images in the dataset WATCHES
	--max-buffer-size: 500000 nb of fetures that will be held in memory before dumping to the database
python index_features.py --dataset datasets/watches/ --features-db features/watchesFeatures.hdf5 --approx-images 3500 --max-buffer-size 500000
WARNING: remove the debug flag in index_features
Duration for 3100 images - +/-50mins

(2)expected results:
------------------
*a database watchesFeatures.hdf5 
*this db stores for each images 
	the keypoints computed by SURF
	the features computed by ROOTSIFT
	the additional data stored, separated by a '#' (reference#title)
(3)how to test:
------------
run the test_feature_extraction.py nd visually make sure the keypoints are corresponding to the image

STEP_2: Create a codebook for the dataset of watches 
=====================================================
(1) run cluster_features.py with following parameters
------------------------------------------------------
	--features-db : features/watchesFeatures.hdf5  (source of the features and keypoints for each images)
	--codebook: codebook/vocab.cpickle (dictionary of visual images - centroids of the clusters of features)
	--clusters: 1536 nb of clusters (nb of words in the codebook)
	--percentage: 0.25 percentage of total features to use to create the cluster
python cluster_features.py --features-db features/watchesFeatures.hdf5 --codebook codebook/vocab.cpickle --clusters 1500 --percentage 0.25
WARNING: remove the debug flag in cluster_features

(2) expected results:
---------------------
A cpickle file in the codebook directory that will contain the centroids of each of clusters of features
duration for 749324 samples:+/-60mins. 835000:85mins

(3) how to test:
----------------
Try to visualize the with following command that will create images in the vocabimages directory
python visualize_centers.py --dataset datasets\watches --features-db features\watchesFeatures.hdf5 --codebook codebook\vocab.cpickle --output vocabimages

STEP_3: Create an histogram for each pictures
=============================================
(1) run extract_bovw with the following parameters
	--features-db: features/watchesFeatures.hdf5  (source of the features and keypoints for each images)
	--codebook: codebook/vocab.cpickle (dictionary of visual images - centroids of the clusters of features)
	--bovw-db: bovw/watchesBovw.hdf5 (where the histograms of each image will be stored)
	--idf: idf.cpickle (not sure what it is yet)
	--max-buffer-size: 500000 (nb of histogram stored in memory before dump to file)
	
python extract_bovw.py --features-db features/watchesFeatures.hdf5 --codebook codebook/vocab.cpickle --bovw-db bovw/watchesBovw.hdf5 --idf idf.cpickle --max-buffer-size 500000
duration: 3100 images in 6mins


(2) expected results:
---------------------
an hdf5 database containing the histograms of features of each images of the dataset

(3) how to test:
----------------
Run this python utility showHistograms.py to show an image and its histogram.(jump to step 4)
Compare with the vocabulary stored in vocabimages directory

STEP_4: find your watch
------------------------
(1) run findMyWatch.py make sure init params are the right ones(bovw, codebook, features) with following prama
	--image: path to the image of a watch for which you need to find similar models
	--nb-bins:700 nb of categories of watches(rule of thumb: use 1/4 of the catalog size)


(2) expected results: your watch is shown as well as all watches from the dataset belonging to same clusters( =similar ones for the computer)