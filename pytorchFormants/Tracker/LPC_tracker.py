from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


dir = "pytorchFormants/Tracker/"

Xtrain_raw = np.load(dir + "LPC_RNN_X_train.npy").astype(np.float32).reshape(-1, 350)
Xtest_raw = np.load(dir + "LPC_RNN_X_test.npy").astype(np.float32).reshape(-1, 350)

Ytrain_raw = np.load(dir + "LPC_RNN_Y_train.npy").astype(np.float32).reshape(-1, 4)
Ytest_raw = np.load(dir + "LPC_RNN_Y_test.npy").astype(np.float32).reshape(-1, 4)
seq = 20
Xtrain = Xtrain_raw[:int(Xtrain_raw.shape[0]/seq)*seq, ].reshape(-1, seq, 350)
Ytrain = Ytrain_raw[:int(Ytrain_raw.shape[0]/seq)*seq, ].reshape(-1, seq, 4)

Xtest = Xtest_raw[:int(Xtest_raw.shape[0]/seq)*seq, ].reshape(-1, seq, 350)
Ytest = Ytest_raw[:int(Ytest_raw.shape[0]/seq)*seq, ].reshape(-1, seq, 4)

class LSTM(nn.Module):

    def __init__(self, input_dim=350, sequence_len=2, hidden_dim=None, batch_size=10, output_dim=1, num_layers=1, device=None):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.sequence_len = sequence_len
        self.device = device

        # Define the LSTM layer
        self.lstm1 = nn.LSTM(input_size=self.input_dim, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256,batch_first=True)
        self.fc = nn.Linear(256, 4)
        self.to(device)

    # def forward(self, x, lengths, h1, h2):
    #     packed_x, sorted_idx = sort_and_pack(x, lengths)
    #     out, _ = self.lstm1(packed_x, h1)
    #     out = unpack_and_unsort(out, sorted_idx)
    #     out = F.sigmoid(out)
    #     out, sorted_idx = sort_and_pack(out, lengths)
    #     out, _ = self.lstm2(out, h2)
    #     out = unpack_and_unsort(out, sorted_idx)
    #     out = self.fc(out)
    #     out = out.view(self.batch_size, -1, 4)
    #     return out

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x  = self.fc(x)
        return x

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
    logits = model.forward(inputs.to())
    return logits.data.cpu().numpy()


loss = nn.L1Loss()
epochs = 100
batchSize = 10

n_batches = int(Xtrain.shape[0]/batchSize)

model = LSTM().to(device)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)


costs = []
test_accuracies = []
print("Starting training ")
for i in range(epochs):
    cost = 0.0
    for j in range(n_batches):
        Xbatch = torch.from_numpy(Xtrain[j*batchSize:(j+1)*batchSize])
        Ybatch = torch.from_numpy(Ytrain[j*batchSize:(j+1)*batchSize])
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
    Ypred = predict(model, torch.from_numpy(Xtest))
    for k in range(0, len(Ytest)):
        for p in range(seq):
            # print(y_hat[i])
            l1 = np.abs(float(Ytest[k, p, 0]) - Ypred[k, p, 0])
            l2 = np.abs(float(Ytest[k, p, 1]) - Ypred[k, p, 1])
            l3 = np.abs(float(Ytest[k, p, 2]) - Ypred[k, p, 2])
            l4 = np.abs(float(Ytest[k, p, 3]) - Ypred[k, p, 3])
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
    loss1 /= len(Ytest)*seq
    loss2 /= len(Ytest)*seq
    loss3 /= len(Ytest)*seq
    loss4 /= len(Ytest)*seq
    total_loss = loss1 + loss2 + loss3 + loss4
    total_loss /= 4.0
    print('median: %.3f %.3f %.3f %.3f' % (np.median(list_1), np.median(list_2), np.median(list_3), np.median(list_4)))
    print('max loss: %.3f %.3f %.3f %.3f' % (max_1, max_2, max_3, max_4))
    print('Real test score: %.3f %.3f %.3f %.3f' % (loss1, loss2, loss3, loss4))
    print("Epoch: %d, acc: %.3f" % (i, total_loss))

    costs.append(cost/n_batches)
    test_accuracies.append(round(total_loss, 3))

    torch.save(model.state_dict(), "LPC_RNN.pt")