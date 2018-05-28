import torch
import torch.utils.data as data
import torch.nn as nn
import loadCIFAR
import matplotlib.pyplot as plt

losses = []
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 10)
        self.LearningRate = 1e-3
        self.epochs = 32
        self.reg = 0

    def forward(self, input):
        layer1 = nn.functional.relu(self.fc1(input))
        layer2 = nn.functional.relu(self.fc2(layer1))
        layer3 = nn.functional.relu(self.fc3(layer2))
        layer4 = self.fc4(layer3)
        return layer4

    def train(self, loader):
        optimizer = torch.optim.SGD(self.parameters(), lr = self.LearningRate, weight_decay = self.reg)
        lossFunction = nn.CrossEntropyLoss()
        for i in range(0, self.epochs):
            for batch in loader:
                input, labels = batch
                optimizer.zero_grad()
                output = net(input)
                loss = lossFunction(output, labels)
                losses.append(loss)
                #print(input)
                #print(output)
                #print(torch.argmax(output), labels)
                print('Loss: ',float(loss), '\n')
                loss.backward()
                optimizer.step()
            print('Epoch {} finished'.format(i))


trainX, trainY, testX, testY = loadCIFAR.loadCIFAR()
trainX, trainY, testX, testY = torch.Tensor(trainX[:4096]), torch.LongTensor(trainY[:4096]), torch.Tensor(testX[:500]), torch.LongTensor(testY[:500])

trainSet = data.TensorDataset(trainX, trainY)
testSet = data.TensorDataset(testX, testY)

trainLoader = data.DataLoader(trainSet, batch_size=32, shuffle=True, num_workers=2)
testLoader = data.DataLoader(testSet, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
net.train(trainLoader)

correct = 0
with torch.no_grad():
    for i in testLoader:
        input, label = i
        output = net(input)
        if (torch.argmax(output) == label):
            correct += 1

print('\nHyperparameters: \n\tLearning rate:\t{}\n\tEpochs:\t\t{}\n\tRegularization:\t{}\n\tFinal Loss:\t{}'.format(net.LearningRate, net.epochs, net.reg,float(losses[-1])))
print('Accuracy trained: {}/{}, {}%\n'.format(correct, testX.shape[0], 100*(correct/testX.shape[0])))
plt.plot(losses)
plt.show()
