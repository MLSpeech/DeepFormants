from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
torch.manual_seed(1)

trainY = np.load("norm_cnn_timit_train_Y.npy")
testY = np.load("norm_cnn_timit_test_Y.npy")
Xtrain = np.load("norm_cnn_timit_train_X.npy").astype(np.float32)
Ytrain = trainY[:,1:5].astype(np.float32)
Xtest = np.load("norm_cnn_timit_test_X.npy").astype(np.float32)
Ytest = testY[:,1:5].astype(np.float32)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
D = Xtrain.shape[1]
K = len(Ytrain)

print(D, K)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(3, 3), stride=1, padding=0)
        self.Conv2 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.Conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=0)
        self.Dense5 = nn.Linear(43*38*64, 512)
        self.out = nn.Linear(512, 4)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.relu(self.Conv3(x))
        x = F.relu(self.Conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        #print(in_size)
        x = x.view(x.size(0), -1)
        x = F.relu(self.Dense5(x))
        return self.out(x)


def train(model, loss, optimizer, inputs, labels):
    inputs = Variable(inputs.to(device))
    labels = Variable(labels.to(device))
    optimizer.zero_grad()

    logits = model.forward(inputs)
    output = loss.forward(logits, labels)
    output.backward()
    optimizer.step()

    return output.item()


def predict(model, inputs):
    inputs = Variable(inputs)
    with torch.no_grad():
        logits = model.forward(inputs.to(device))
    return logits.data.cpu().numpy()


Xtrain = torch.from_numpy(Xtrain).float().to(device)
Ytrain = torch.from_numpy(Ytrain).float().to(device)
Xtest = torch.from_numpy(Xtest).float().to(device)
Ytest = torch.from_numpy(Ytest).float().to(device)


model = Net().to(device)
loss = nn.L1Loss()
optimizer = optim.Adagrad(model.parameters())

epochs = 80
batchSize = 32
n_batches = int(np.floor(Xtrain.size()[0]/batchSize))

costs = []
test_accuracies = []
print("Starting training ")
for i in range(epochs):
    cost = 0.0
    for j in range(n_batches):
        #print(j, '/', n_batches)
        Xbatch = Xtrain[j*batchSize:(j+1)*batchSize]
        Ybatch = Ytrain[j*batchSize:(j+1)*batchSize]
        cost += train(model, loss, optimizer, Xbatch, Ybatch)

    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss4 = 0.0
    max_1 = 0.0
    max_2 = 0.0
    max_3 = 0.0
    max_4 = 0.0
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    print('predicting...')
    Ypred = predict(model, Xtest)
    for k in range(0, len(Ytest)):
        # print(y_hat[i])
        l1 = np.abs(float(Ytest[k, 0]) - Ypred[k, 0])
        l2 = np.abs(float(Ytest[k, 1]) - Ypred[k, 1])
        l3 = np.abs(float(Ytest[k, 2]) - Ypred[k, 2])
        l4 = np.abs(float(Ytest[k, 3]) - Ypred[k, 3])
        list_1.append(l1)
        list_2.append(l2)
        list_3.append(l3)
        list_4.append(l4)
        max_1 = max(max_1, l1)
        max_2 = max(max_2, l2)
        max_3 = max(max_3, l3)
        max_4 = max(max_4, l4)
        loss1 += l1
        loss2 += l2
        loss3 += l3
        loss4 += l4
    loss1 /= len(Ytest)
    loss2 /= len(Ytest)
    loss3 /= len(Ytest)
    loss4 /= len(Ytest)
    total_loss = loss1 + loss2 + loss3 + loss4
    total_loss /= 4.0
    print('median: %.3f %.3f %.3f %.3f' % (np.median(list_1), np.median(list_2), np.median(list_3), np.median(list_4)))
    print('max loss: %.3f %.3f %.3f %.3f' % (max_1, max_2, max_3, max_4))
    print('Real test score: %.3f %.3f %.3f %.3f' % (loss1, loss2, loss3, loss4))
    print("Epoch: %d, acc: %.3f" % (i, total_loss))

    costs.append(cost/n_batches)
    test_accuracies.append(round(total_loss, 3))
    torch.save(model.state_dict(), "CNN_estimate.pt")

print(test_accuracies)