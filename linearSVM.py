import numpy as np
from loadCIFAR import loadCIFAR

class linearSVM():

	# class is responsible for storing training data set & labels
	# and all hyperparamaters
	def __init__(self, X, Y, W, d=1.0):
		self.Xtr = X
		self.Ytr = Y
		self.W = W
		self.delta = d
		self.gradients = np.zeros(self.W.shape)

	# uses gradient descent to optimize loss function with regards to weight
	# X is training images, Y is training labels
	def train(self, X, Y):
		step_size = 1 ** -6 
		print('Training...\n')
		for i in range(X.shape[0]):
			weight_gradients = self.numerical_gradient(X[i], Y[i])
			self.W += -weight_gradients * step_size
			print(self.W)

	# approximates gradient of loss function w/ respect to weights
	# using numerical method f'(x) = ( f(x+h) - f(x) ) / h
	def numerical_gradient(self, X , Y):
		fx = self.loss_i(X, Y)
		h = .00001
		self.gradients = np.zeros(self.W.shape)
		for j in range(self.gradients.shape[1]):
			for i in range(self.gradients.shape[0]):
				oldW = self.W[i][j]
				self.W[i][j] = oldW + h
				fxh = self.loss_i(X, Y)
				self.gradients[i][j] = (fxh - fx) / h
				self.W[i][j] = oldW
		return self.gradients
			
	# finds gradient of loss function w/ respect to weights using derivative
	# of loss function found w/ calculus
	def gradient(self, X, Y):
		for i in range(self.gradients.shape[0]):
			if i == Y:
				sumOutOfMargin = 0
				for j in range(self.gradients.shape[0]):
					if(self.loss_i(X,Y) > 0):
						sumOutOfMargin += 1
				self.gradients[i] = sumOutOfMargin*-X
			else:
				if(self.loss_i(X,Y) > 0):
					self.gradients[i] = X
				else:
					self.gradients[i] = 0
		return self.gradients

	# returns SVM loss of X training image and Y training label
	def loss_i(self, X, Y):
		scores = self.W.dot(X)
		trueScore = scores[Y]
		loss = np.maximum(0, scores - trueScore + self.delta)
		loss[Y] = 0
		return np.sum(loss) 

	# L2 regularization to prevent overfitting
	def regularize(self):
		return np.sum(self.W * self.W)

	# returns loss over entire data set
	def lossTotal(self, X, Y):
		sumLoss = 0
		for i in range(X.shape[0]):
			sumLoss += self.loss_i(X[i],Y[i])
		return sumLoss/X.shape[0] + self.regularize()			

	# returns scores of classes in X image
	def score(self, X):
		scores = self.W.dot(X)
		return scores 
	
	# returns highest score of classes in X image, aka predicted class
	def evaluate(self, X):
		scores = self.score(X)
		return scores.argsort()[-1]

def main():
	trainingData, trainingLabels, testData, testLabels = loadCIFAR()
	weights = np.random.rand(10, 3072) * .001
	clf = linearSVM(trainingData, trainingLabels, weights)
	test_set = testData[:100]
	trainingData = trainingData[:512]
	untrainedCorrect = 0
	trainedCorrect = 0
	trainingEpochs = 10

	for i in range(trainingData.shape[0]):
		prediction = clf.evaluate(trainingData[i])
		if( prediction == trainingLabels[i]):
			untrainedCorrect += 1
	
	for i in range(trainingEpochs):
		clf.train(trainingData, trainingLabels)

	for i in range(test_set.shape[0]):
		prediction = clf.evaluate(test_set[i])
		if( prediction == testLabels[i]):
			trainedCorrect += 1
	
	print('\nAccuracy untrained: {}%'.format(untrainedCorrect/trainingData.shape[0] * 100))
	print('Accuracy trained: {}%\n'.format(trainedCorrect/test_set.shape[0] * 100))

if (__name__ == '__main__'):
	main()
