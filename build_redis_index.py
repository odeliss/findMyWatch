# import the necessary packages
from __future__ import print_function
from pyimagesearch.db.redisqueue import RedisQueue
from redis import Redis
import argparse
import h5py
from pathlib import Path

DEBUG = True
REDIS_PORT = 6379 

# For each bovw item, this will store in redis DB a list of imageIDs containing the bovw item
# construct the argument parser and parse the arguments
if DEBUG == False:
	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--bovw-db", required=True, help="Path to where the bag-of-visual-words database")
	args = vars(ap.parse_args())
	bovwDB = args["bovw_db"]
else:
	bovwDB = str(Path("testData/bovw/watchesBovw.hdf5"))

# connect to redis, initialize the redis queue, and open the bag-of-visual-words database
redisDB = Redis(host="localhost", port=REDIS_PORT, db=0, charset="utf-8", decode_responses=True)
rq = RedisQueue(redisDB)
bovwDB = h5py.File(bovwDB, mode="r")

# loop over the entries in the bag-of-visual-words
for (i, hist) in enumerate(bovwDB["bovw"]):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		print("[PROGRESS] processed {} entries".format(i))

	# add the image index and histogram to the redis server
	rq.add(i, hist)

# close the bag-of-visual-words database and finish the indexing processing
bovwDB.close()
rq.finish()
