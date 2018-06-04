import torch
import torch.utils.data as data
import torch.nn as nn
import loadCIFAR
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self,lr, reg):
        super(Net,self).__init__()

        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 10)
        self.LearningRate = lr
        self.epochs = 16
        self.reg = reg

    def forward(self, input):
        layer1 = nn.functional.relu(self.fc1(input))
        layer2 = nn.functional.relu(self.fc2(layer1))
        layer3 = nn.functional.relu(self.fc3(layer2))
        layer4 = self.fc4(layer3)
        return layer4

    def train(self, loader):
        losses = np.zeros(len(loader))
        optimizer = torch.optim.SGD(self.parameters(), lr = self.LearningRate, weight_decay = self.reg, momentum = .9)
        lossFunction = nn.CrossEntropyLoss()
        for i in range(0, self.epochs):
            for batch in loader:
                input, labels = batch
                optimizer.zero_grad()
                output = net(input)
                loss = lossFunction(output, labels)
                print('Loss: ', loss)
                losses[i] = loss
                loss.backward()
                optimizer.step()
        plt.plot(losses)
        plt.show()
        return loss
    
    def test(self, loader):
        print(len(loader))
        correct = 0
        with torch.no_grad():
            for i in testLoader:
                input, label = i
                output = net(input)
                if (torch.argmax(output) == label):
                    correct += 1
        accuracy = 100 * (correct / len(loader))
        print('\nHyperparameters: \n\tLearning rate:\t{}\n\tRegularization:\t{}'.format(self.LearningRate, self.reg))
        print('Accuracy trained: {}/{}, {}%\n'.format(correct, len(loader), accuracy))
        return accuracy



trainX, trainY, testX, testY = loadCIFAR.loadCIFAR()
trainX, trainY, testX, testY = torch.Tensor(trainX[:30000]), torch.LongTensor(trainY[:30000]), torch.Tensor(testX[:500]), torch.LongTensor(testY[:500])

trainSet = data.TensorDataset(trainX, trainY)
testSet = data.TensorDataset(testX, testY)

trainLoader = data.DataLoader(trainSet, batch_size=32, shuffle=True, num_workers=2)
testLoader = data.DataLoader(testSet, batch_size=1, shuffle=False, num_workers=2)

def gridSearch():
    learningRange = range(-10, -1)
    regRange = range(-10,-1)
    accuracies = np.zeros([len(learningRange),len(regRange)])
    for lr in learningRange:
        for reg in regRange:
            net = Net(10**(lr), 10**(reg))
            currentLoss = net.train(trainLoader)
            currentAccuracy = net.test(testLoader)
            accuracyIndex = (regRange.index(reg), learningRange.index(lr))
            accuracies[accuracyIndex] = currentAccuracy
            del net
    print(accuracies)

net = Net(lr = 1e-5, reg = 1e-3)
net.train(trainLoader)
net.test(testLoader)
