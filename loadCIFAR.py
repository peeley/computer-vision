import pickle
import numpy as np

#
# returns training data, labels, then test data, and labels
#

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def loadCIFAR():
	trainingData = []
	trainingLabels = []
	for i in range(1,5):
		trainPath = 'cifar-10-batches-py/data_batch_%i'%i
		rawData = unpickle(trainPath)
		trainingData.append(rawData[b'data'])
		trainingLabels.append(rawData[b'labels'])
	trainingData = np.concatenate(trainingData)
	trainingLabels = np.concatenate(trainingLabels)
	testPath = 'cifar-10-batches-py/test_batch'
	rawTest = unpickle(testPath)
	testData = rawTest[b'data']
	testLabels = rawTest[b'labels']
	return trainingData, trainingLabels, testData, testLabels

