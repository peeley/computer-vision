import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from loadCIFAR import loadCIFAR

class knnClassifier:
	# initializer, X is array of training data, y is labels, k is # of neighbors
	def __init__(self, X, Y, k=1):
		self.Xtrain = X
		self.Ytrain = Y
		self.k = k
	# evaluates L2 distance between parameter testImage and all other images
	# in training data, then returns label of closest images
	def evaluateImage(self, testImage):
		# if there are no neighbors, no label is closest!
		if self.k == 0:
			return -1
		# loop through all training data, compute distance from input testImage
		dists = np.zeros(testImage.shape[0])
		for i in range(dists.shape[0]):
			dists[i] = L1dist(self.Xtrain[i,:], testImage)
		# find the index of the k closest in dists
		kClosestLocs = dists.argsort()[:self.k]
		clfLabels = np.zeros(self.k)
		# get the labels of the k closest
		for i in range(kClosestLocs.shape[0]):
			clfLabels[i] = self.Ytrain[kClosestLocs[i]]
		# find and return most common label among closest
		count = Counter(clfLabels)
		return count.most_common()[0][0]

def L1dist(train, test):
	return int(np.sum(np.abs(test - train)))

def L2dist(test, train):
	return int(np.sqrt(np.sum(np.square(test - train))))

if __name__ == '__main__':

	trainingImages, trainingLabels, testImages, testLabels = loadCIFAR()

	kTests = np.arange(20)
	kResults = np.zeros(kTests.shape[0], dtype=float)
	for k in kTests:
		clf = knnClassifier(trainingImages, trainingLabels, kTests[k])
		correct = 0
		batchSize = 500
		for i in range(batchSize):
			clfVal = clf.evaluateImage(testImages[i])
			if(clfVal == testLabels[i]):
				correct += 1
		acc = float(correct/batchSize) * 100
		print('{}% acc, batch size {}, k={}'.format((correct/batchSize)*100, batchSize,clf.k))
		kResults[k] = acc
	bestKLoc = np.argmax(kResults)
	print("\nBest accuracy of {}% at k={}".format(kResults[bestKLoc], kTests[bestKLoc]))
	plt.plot(kTests, kResults) 
	plt.ylabel("Accuracy (%)")
	plt.xlabel("k-Nearest Neighbors")
	plt.title("Accuracy of kNN model vs. k-nearest neighbors using L1 distance")
	fig = plt.figure()
	fig.savefig('performance.jpg',dpi = fig.dpi)
	plt.show()

