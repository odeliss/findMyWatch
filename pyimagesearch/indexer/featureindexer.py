# import the necessary packages
from pyimagesearch.indexer.baseindexer import BaseIndexer
import numpy as np
import h5py
#[12-11-2016] added additional data in the HDF5 DB
#[28-11-2016] added title as new scraped data
#[26-12-2016] added option to append data to an existing feature file

class FeatureIndexer(BaseIndexer): #is inherited from the Baseindexer

	def __init__(self, dbPath, appendSwitch, estNumImages=500, maxBufferSize=50000, dbResizeFactor=2,
		verbose=True ): #[26-12-2016 +++] append boolean added if data needs to be added to file
                # call the parent constructor
                #print(dbPath)
                super(FeatureIndexer, self).__init__(dbPath, estNumImages=estNumImages,
                        maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor,
                        verbose=verbose)

                
                
                #[26-12-2016 +++] initialize the db dataset with existing set if append and none if new
                if appendSwitch == True:
                # open the HDF5 database for writing and initialize the datasets within
                # the group
                        self.db = h5py.File(self.dbPath, mode="a") #[26-12-2016 +++] changed mode from w to a
                
                        self.imageIDDB = self.db["image_ids"]
                        self.indexDB = self.db["index"]
                        self.featuresDB = self.db["features"]
                        self.dataDB = self.db["addData"]

                        nbImagesIdx = self.db["image_ids"].shape[0] #Select nb image_ids https://gurus.pyimagesearch.com/lessons/extracting-keypoints-and-local-invariant-descriptors/
                        nbFeaturesIdx = self.db["features"].shape[0] #Select nb of rows in the image_ids. Same reference
                        self.idxs = {"index": nbImagesIdx, "features": nbFeaturesIdx}


                else:
                # open the HDF5 database for writing and initialize the datasets within
                # the group
                        self.db = h5py.File(self.dbPath, mode="w") 
                
                        self.imageIDDB = None
                        self.indexDB = None
                        self.featuresDB = None
                        self.dataDB = None #+++[12-11-2016] - added additional data field

                        self.idxs = {"index": 0, "features": 0}
                #[26-12-2016 ---]
                        
                # initialize the image IDs buffer, index buffer and the keypoints +
                # features buffer
                self.imageIDBuffer = []
                self.indexBuffer = []
                self.featuresBuffer = None
                self.dataBuffer = [] #+++[12-11-2016] - added additional data field

                # initialize the total number of features in the buffer along with the
                # indexes dictionary
                self.totalFeatures = 0
                #+++ [12-11-2016] added additional data in the HDF5 DB (affiliate link)
	def add(self, imageID, kps, features, additionalData):
		# compute the starting and ending index for the features lookup
		# the additional data is the ASIN code, and the title provided  
		# as an additionalData list and stored in the DB
		start = self.idxs["features"] + self.totalFeatures
		end = start + len(features)

		# update the image IDs buffer, features buffer, and index buffer,
		# followed by incrementing the feature count
		self.imageIDBuffer.append(imageID)
		self.featuresBuffer = BaseIndexer.featureStack(np.hstack([kps, features]),
			self.featuresBuffer)
		self.indexBuffer.append((start, end))
		self.dataBuffer.append(additionalData)#+++[12-11-2016] added additional data in the HDF5 DB (affiliate link)
		
		self.totalFeatures += len(features)

		# check to see if we have reached the maximum buffer size
		if self.totalFeatures >= self.maxBufferSize:
			# if the databases have not been created yet, create them
			if None in (self.imageIDDB, self.indexDB, self.featuresDB, self.dataDB):#+++[12-11-2016] added additional data in the HDF5 DB
				self._debug("initial buffer full")
				self._createDatasets()

			# write the buffers to file
			self._writeBuffers()
	
	def _createDatasets(self):
		# compute the average number of features extracted from the initial buffer
		# and use this number to determine the approximate number of features for
		# the entire dataset
		avgFeatures = self.totalFeatures / float(len(self.imageIDBuffer))
		approxFeatures = int(avgFeatures * self.estNumImages)

		# grab the feature vector size
		fvectorSize = self.featuresBuffer.shape[1]

		# initialize the datasets
		self._debug("creating datasets...")
		self.imageIDDB = self.db.create_dataset("image_ids", (self.estNumImages,),
			maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
		self.indexDB = self.db.create_dataset("index", (self.estNumImages, 2),
			maxshape=(None, 2), dtype="int")
		self.featuresDB = self.db.create_dataset("features",
			(approxFeatures, fvectorSize), maxshape=(None, fvectorSize),
			dtype="float")
		#+++[12-11-2016] - added additional data field containing only 1 data - the affiliate URL
		self.dataDB = self.db.create_dataset("addData", (self.estNumImages,),
			maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
		#---[12-11-2016] - added additional data ASIN field	

	def _writeBuffers(self):
		# write the buffers to disk
		self._writeBuffer(self.imageIDDB, "image_ids", self.imageIDBuffer,
			"index")
		self._writeBuffer(self.indexDB, "index", self.indexBuffer, "index")
		self._writeBuffer(self.featuresDB, "features", self.featuresBuffer,
			"features")
		#+++[12-11-2016] added additional data in the HDF5 DB (affiliate link)
		self._writeBuffer(self.dataDB, "addData", self.dataBuffer, "index")
		#---[12-11-2016] added additional data in the HDF5 DB (affiliate link)
		# increment the indexes
		self.idxs["index"] += len(self.imageIDBuffer)
		self.idxs["features"] += self.totalFeatures

		# reset the buffers and feature counts
		self.imageIDBuffer = []
		self.indexBuffer = []
		self.featuresBuffer = None
		self.dataBuffer = [] #+++[12-11-2016] added additional data in the HDF5 DB (affiliate link)
		self.totalFeatures = 0

	def finish(self):
		# if the databases have not been initialized, then the original
		# buffers were never filled up
		if None in (self.imageIDDB, self.indexDB, self.featuresDB, self.dataDB):#+++[12-11-2016] added additional data in the HDF5 DB (affiliate link)
			self._debug("minimum init buffer not reached", msgType="[WARN]")
			self._createDatasets()

		# write any unempty buffers to file
		self._debug("writing un-empty buffers...")
		self._writeBuffers()

		# compact datasets
		self._debug("compacting datasets...")
		self._resizeDataset(self.imageIDDB, "image_ids", finished=self.idxs["index"])
		self._resizeDataset(self.indexDB, "index", finished=self.idxs["index"])
		self._resizeDataset(self.featuresDB, "features", finished=self.idxs["features"])
		self._resizeDataset(self.dataDB, "addData", finished=self.idxs["index"])#+++[12-11-2016] added additional data in the HDF5 DB (affiliate link)
		
		# close the database
		self.db.close()
