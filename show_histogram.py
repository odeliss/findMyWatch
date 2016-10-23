import h5py
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

featuresDB = h5py.File("features/watchesFeatures.hdf5")
bovwDB = h5py.File("bovw/watchesBovw.hdf5")
data=[]
length=[]
NB_BINS = 800

for (i,imageID) in enumerate(bovwDB["bovw"]):
    hist = bovwDB["bovw"][i]
    data.append(hist)


# cluster the color histograms
print("[INFO] Clustering the data...")
clt = KMeans(n_clusters=NB_BINS)
labels = clt.fit_predict(data)



# plot the distribution of the watches per categories
for bins in np.unique(labels):
    length.append(len(np.where(labels == bins)[0]))
plt.plot(np.unique(labels),length)
plt.axis([0,NB_BINS,0,100])
plt.show()
print("[INFO] plot ready...")   

# loop over the unique labels
for label in np.unique(labels):
        # grab all image paths that are assigned to the current label
        labelPathsIDs = np.where(labels == label)
        
        # loop over the image paths that belong to the current label
        for ID in labelPathsIDs[0]:
                # load the image and display it
                imageFile="datasets//"+str(featuresDB["image_ids"][ID])
                image = cv2.imread(imageFile)
                cv2.imshow("Cluster {}, Image #{}".format(label + 1, ID + 1), image)
                cv2.waitKey(0)
                # wait for a keypress and then close all open windows
        cv2.destroyAllWindows()
