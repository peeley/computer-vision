import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12*5*5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.learningRate = .01
        self.reg = 1e-6
        self.epochs = 16

    def forward(self, input):
        x = nn.functional.relu(self.conv1(input))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 12 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, loader):
        optimizer = torch.optim.SGD(self.parameters(), lr = self.learningRate, weight_decay = self.reg, momentum = .9)
        loss_fn = torch.nn.CrossEntropyLoss()
        losses = np.zeros(len(loader) * self.epochs)
        counter = 0
        for i in range(self.epochs):
            for batch in loader:
                input, labels = batch
                optimizer.zero_grad()
                output = self.forward(input)
                loss = loss_fn( output, labels)
                print('Loss: ', loss)
                losses[counter] = loss
                counter += 1
                loss.backward()
                optimizer.step()
            print('\nEpoch {} finished \n'.format(i + 1))
        plt.plot(losses)
        plt.show()


net = convNet()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
net.train(trainloader)
correct = 0
items = 0
for i in testloader:
    input, labels = i
    items += 1
    output = torch.argmax(net(input))
    if output == labels:
        correct += 1
print('Accuracy: {}/{}, {}%'.format(correct, items, (correct/items)*100))
