import numpy as np
from loadCIFAR import loadCIFAR
import matplotlib.pyplot as plt
class linearSVM():

	# class is responsible for storing training data set & labels
	# and all hyperparamaters
	def __init__(self, X, Y, W, reg=1e3, learning_rate=1e-7):
		self.Xtr = X
		self.Ytr = Y
		self.W = W
		self.reg = reg
		self.learning_rate = learning_rate

	# uses gradient descent to optimize SVM loss function with regards to weight
	# X is training images, Y is training labels
	def svm_train(self, X, Y, batchSize):
		randLoc = np.random.choice(X.shape[0], batchSize, replace=True)
		batchData = X[randLoc]
		batchLabels = Y[randLoc]
		loss, weight_gradients = self.svm_gradient(batchData, batchLabels)
		self.W += weight_gradients * -self.learning_rate
		print('Loss: {}\n'.format(loss))
		return loss
	
	# uses gradient descent to optimize cross-entropy loss function with regards
	# to weight, X is training batch images & Y is batch labels
	def softmax_train(self, X, Y, batchSize):
		randLoc = np.random.choice(X.shape[0], batchSize, replace=True)
		batchData = X[randLoc]
		batchLabels = Y[randLoc]
		loss, weight_gradients = self.softmax_gradient(batchData, batchLabels)
		self.W += weight_gradients * -self.learning_rate
		print('Loss: {}\n'.format(loss))
		return loss

	# returns loss over entire data set and gradient, X is all training images
	# and Y is training labels
	def svm_gradient(self, X, Y):
		scores = X.dot(self.W) 
		trueScores = scores[np.arange(scores.shape[0]), Y]
		margins = np.maximum(0, scores - np.matrix(trueScores).T + 1)
		margins[np.arange(X.shape[0]), Y] = 0
		loss = np.mean(np.sum(margins, axis=1))
		loss += .5 * self.reg * np.sum(self.W * self.W)
		binary = margins
		binary[margins > 0] = 1
		row_sum = np.sum(margins, axis=1)
		binary[np.arange(X.shape[0]), Y] = -row_sum.T
		dW = np.dot(X.T, binary)
		dW /= X.shape[0]
		dW += self.reg * self.W
		return loss, dW

	# approximates gradient of loss function w/ respect to weights
	# using numerical method f'(x) = ( f(x+h) - f(x) ) / h
	def svm_numerical_gradient(self, X , Y):
		fx = self.loss_i(X, Y)
		h = 1e-5
		gradients = np.zeros(self.W.shape)
		for j in range(gradients.shape[1]):
			for i in range(gradients.shape[0]):
				oldW = self.W[i][j]
				self.W[i][j] = oldW + h
				fxh = self.loss_i(X, Y)
				gradients[i][j] = (fxh - fx) / h
				self.W[i][j] = oldW
		return gradients

	# finds loss and gradient of cross-entropy loss function for softmax
	# classifier
	def softmax_gradient(self, X, Y):
		X1 = X.dot(self.W)
		scores = np.exp(X1)
		scores = (scores.T/np.sum(scores, axis =1)).T
		ones = np.zeros([X.shape[0], self.W.shape[1]])
		for i, row in enumerate(ones):
			row[Y[i]] = 1
		loss_ma = np.ma.log(scores * ones)
		loss = np.sum(-loss_ma.filled(0))/X.shape[0] +.05*self.reg*np.sum(self.W + self.W)
		dX1 = scores - ones
		dW = ((dX1.T).dot(X)).T/X.shape[0] + self.reg*self.W
		return loss, dW

	# returns SVM loss of X training image and Y training label
	def loss_i(self, X, Y):
		scores = X.dot(self.W)
		trueScore = scores[Y]
		loss = np.maximum(0, scores - trueScore + 1)
		loss[Y] = 0
		return np.sum(loss) 

	# L2 regularization to prevent overfitting
	def regularize(self):
		return np.sum(self.W * self.W)


	# returns scores of classes in X image
	def score(self, X):
		scores = X.dot(self.W)
		return scores 
	
	# returns highest score of classes in X image, aka predicted class
	def evaluate(self, X):
		scores = self.score(X)
		return scores.argsort()[-1]

	def softmax(self, X):
		scores = self.score(X)
		probs = (np.exp(scores)/np.sum(np.exp(scores))) * 100
		return np.argmax(probs)

def main():
	trainingData, trainingLabels, testData, testLabels = loadCIFAR()
	validationData, validationLabels = trainingData[30000:], trainingLabels[30000:]
	trainingData, trainingLabels = trainingData[:30000], trainingLabels[:30000]
	weights = np.random.rand(3072, 10) * .00001
	clf = linearSVM(trainingData, trainingLabels, weights)
	batchSize = 256
	untrainedCorrect = 0
	trainedCorrect = 0
	trainingEpochs = 5000
	trainingLosses = np.zeros(trainingEpochs)
	learningRates = [1e1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-10]
	regs = [1e5, 1e3, 1e1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-10]
	validationAccs = np.zeros((len(learningRates),len(regs)))

	untrainedLoss, untrainedGradient = clf.softmax_gradient(trainingData, trainingLabels)
	for i in range(trainingData.shape[0]):
		prediction = clf.softmax(trainingData[i])
		if( prediction == trainingLabels[i]):
			untrainedCorrect += 1

	for i in range(trainingEpochs):
		print('Training epoch {}:'.format(i))
		trainingLosses[i] = clf.softmax_train(trainingData, trainingLabels, batchSize)
	
	trainedLoss, trainedGradient = clf.softmax_gradient(trainingData, trainingLabels)
	for i in range(testData.shape[0]):
		prediction = clf.softmax(testData[i])
		if( prediction == testLabels[i]):
			trainedCorrect += 1
	
	
	print('\nAccuracy untrained: {}%'.format(untrainedCorrect/trainingData.shape[0] * 100))
	print('Untrained loss: {}'.format(untrainedLoss))
	print('Accuracy trained: {}%'.format(trainedCorrect/testData.shape[0] * 100))
	print('Trained loss: {}\n'.format(trainedLoss))

	plt.plot(np.arange(trainingEpochs), trainingLosses)
	plt.show()

if (__name__ == '__main__'):
	main()
