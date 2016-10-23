# import the necessary packages
from __future__ import print_function
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import datetime
import h5py
import progressbar

class Vocabulary:
	def __init__(self, dbPath, verbose=True):
		# store the database path and the verbosity setting
		self.dbPath = dbPath
		self.verbose = verbose

	def fit(self, numClusters, samplePercent, randomState=None):
		# open the database and grab the total number of features
		db = h5py.File(self.dbPath)
		totalFeatures = db["features"].shape[0]

		# determine the number of features to sample, generate the indexes of the
		# sample, sorting them in ascending order to speedup access time from the
		# HDF5 database
		sampleSize = int(np.ceil(samplePercent * totalFeatures))
		idxs = np.random.choice(np.arange(0, totalFeatures), (sampleSize), replace=False)
		idxs.sort()
		data = []
		self._debug("starting sampling...{} samples".format(len(idxs)))

		# loop over the randomly sampled indexes and accumulate the features to
		# cluster
		widgets = ["Indexing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
		#pbar=progressbar.ProgressBar(maxval=len(idxs), widgets=widgets)
		#pbar.start()
		for i in idxs:
			data.append(db["features"][i][2:])
			#pbar.update(i)
			#print(int(i/len(idxs)))
		#pbar.finish()

		# cluster the data
		self._debug("sampled {:,} features from a population of {:,}".format(
			len(idxs), totalFeatures))
		self._debug("clustering with k={:,}".format(numClusters))
		clt = MiniBatchKMeans(n_clusters=numClusters, random_state=randomState)
		clt.fit(data)
		self._debug("cluster shape: {}".format(clt.cluster_centers_.shape))

		# close the database
		db.close()

		# return the cluster centroids
		return clt.cluster_centers_

	def _debug(self, msg, msgType="[INFO]"):
		# check to see the message should be printed
		if self.verbose:
			print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))
