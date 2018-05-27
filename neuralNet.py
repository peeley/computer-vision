import torch
import torch.utils.data as data
import torch.nn as nn
import loadCIFAR

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(3072, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.LearningRate = .001
        self.epochs = 8

    def forward(self, input):
        layer1 = nn.functional.relu(self.fc1(input))
        layer2 = nn.functional.relu(self.fc2(layer1))
        layer3 = self.fc3(layer2)
        return layer3

    def train(self, loader):
        optimizer = torch.optim.SGD(self.parameters(), lr = self.LearningRate, weight_decay = 1e-5)
        lossFunction = nn.CrossEntropyLoss()
        for i in range(0, self.epochs):
            for batch in loader:
                input, labels = batch
                optimizer.zero_grad()
                output = net(input)
                loss = lossFunction(output, labels)
                print(input)
                print(output)
                print(torch.argmax(output), labels)
                print('Loss: ',float(loss), '\n')
                loss.backward()
                optimizer.step()
            print('Epoch {} finished'.format(i))


trainX, trainY, testX, testY = loadCIFAR.loadCIFAR()
trainX, trainY, testX, testY = torch.Tensor(trainX[:100]), torch.LongTensor(trainY[:100]), torch.Tensor(testX[:500]), torch.LongTensor(testY[:500])

trainSet = data.TensorDataset(trainX, trainY)
testSet = data.TensorDataset(testX, testY)

trainLoader = data.DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=2)
testLoader = data.DataLoader(testSet, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
net.train(trainLoader)

correct = 0
with torch.no_grad():
    for i in testLoader:
        input, label = i
        output = net(input)
        if (int(torch.argmax(output)) == int(label)):
            correct += 1

print('Accuracy trained: {}/{}, {}%'.format(correct, testX.shape[0], 100*(correct/testX.shape[0])))
