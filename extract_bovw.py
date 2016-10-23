# USAGE
# python extract_bovw.py --features-db output/features.hdf5 --codebook output/vocab.cpickle \
#	--bovw-db output/bovw.hdf5 --idf output/idf.cpickle

# import the necessary packages
from pyimagesearch.ir import BagOfVisualWords
from pyimagesearch.indexer import BOVWIndexer
import argparse
import _pickle as cPickle
import h5py

DEBUG = False

if DEBUG == False:
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-f", "--features-db", required=True,
                help="Path the features database")
        ap.add_argument("-c", "--codebook", required=True,
                help="Path to the codebook")
        ap.add_argument("-b", "--bovw-db", required=True,
                help="Path to where the bag-of-visual-words database will be stored")
        ap.add_argument("-d", "--idf", required=True,
                help="Path to inverse document frequency counts will be stored")
        ap.add_argument("-s", "--max-buffer-size", type=int, default=500,
                help="Maximum buffer size for # of features to be stored in memory")
        args = vars(ap.parse_args())
        codebook=args["codebook"]
        features_db=args["features_db"]
        bovw_db=args["bovw_db"]
        maxBufferSize=args["max_buffer_size"]
        idf=args["idf"]
else:
        codebook="codebook\\vocab.cpickle"
        features_db="features\\featuresquiz.hdf5"
        bovw_db="output\\bowv.hdf5"
        maxBufferSize=1000
        idf="output\\idf.pickle"


# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = cPickle.loads(open(codebook, "rb").read())
bovw = BagOfVisualWords(vocab)

# open the features database and initialize the bag-of-visual-words indexer
featuresDB = h5py.File(features_db, mode="r")
bi = BOVWIndexer(bovw.codebook.shape[0], bovw_db,
	estNumImages=featuresDB["image_ids"].shape[0],
	maxBufferSize=maxBufferSize)

# loop over the image IDs and index
for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		bi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	# extract the feature vectors for the current image using the starting and
	# ending offsets (while ignoring the keypoints) and then quantize the
	# features to construct the bag-of-visual-words histogram
	features = featuresDB["features"][offset[0]:offset[1]][:, 2:]
	hist = bovw.describe(features)

	# add the bag-of-visual-words to the index
	bi.add(hist)

# close the features database and finish the indexing process
featuresDB.close()
bi.finish()

# dump the inverse document frequency counts to file
f = open(idf, "wb")
f.write(cPickle.dumps(bi.df(method="idf")))
f.close()
