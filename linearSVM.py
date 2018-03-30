import numpy as np
from loadCIFAR import loadCIFAR
class linearSVM():
	def __init__(self, X, Y, W, d=1.0):
		self.Xtr = X
		self.Ytr = Y
		self.W = W
		self.delta = d

	# uses gradient descent to optimize loss function with regards to weight
	def train():
		# TODO
		pass

	# returns SVM loss of X training image and Y training label
	def loss_i(self, X, Y):
		scores = self.W.dot(X)
		trueScore = scores[Y]
		losses = np.maximum(0, scores - scores[Y] + self.delta)
		losses[Y] = 0
		return np.sum(losses)

	# returns highest score of classes in X image
	def evaluate(self, X):
		scores = self.W.dot(X)
		return scores.argsort()[-1]

def main():
	trainingData, trainingLabels, testData, testLabels = loadCIFAR()
	weights = np.random.randint(5, size=[10, trainingData[0].shape[0]])
	clf = linearSVM(trainingData, trainingLabels, weights)
	for i in range(10):
		clfVal = clf.evaluate(testData[i])
		print('CLF: {}'.format( clfVal))
		print('True: {}'.format(testLabels[i]))
		print('Loss: {}\n'.format(clf.loss_i(testData[i], testLabels[i])))

if (__name__ == '__main__'):
	main()
