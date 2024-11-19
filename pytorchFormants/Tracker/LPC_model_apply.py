import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import os
import csv
import math
from collections import defaultdict
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
import wandb

run = wandb.init(project="asr-formants")


def file_to_array(f_name):
    with open(f_name, 'r') as x:
        data = list(csv.reader(x, delimiter="\t"))
    return np.array(data)


class LSTM(nn.Module):

    def __init__(self, input_dim=350, sequence_len=2, hidden_dim=None, batch_size=20, output_dim=1, num_layers=1, device=None):
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

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x


def predict(model, inputs):
    inputs = Variable(inputs)
    logits = model.forward(inputs.to())
    return logits.data.cpu().numpy()


def get_Y_by_file(f_name):
    mask = np.where(np.any(np.isin(Ytest_raw, f_name), axis=1))
    X_test = Xtest_raw[mask, 1:].astype(np.float32).reshape(-1, 350)
    Y_test = Ytest_raw[mask, 1:].astype(np.float32).reshape(-1, 4)
    Y_pred = np.zeros(Y_test.shape)
    for i in range(X_test.shape[0]):
        y_ = X_test[i].reshape(1, -1)
        Y_pred[i] = predict(model, torch.from_numpy(y_))
    return Y_test, np.round(Y_pred, 3)


def get_y_by_phone(labels, ytest, ypred, phone_class: list):
    vow_mask = np.where(np.any(np.isin(labels, phone_class), axis=1))
    y_actual = ytest[vow_mask, :].astype(np.float32)
    y_pred = ypred[vow_mask, :]
    return y_actual, y_pred


def get_error(y_actual, y_predicted):
    y_actual, y_predicted = y_actual.reshape(y_actual.shape[1:]), y_predicted.reshape(y_predicted.shape[1:])
    mean_abs_dif_ = np.mean(np.abs(np.subtract(y_actual, y_predicted)), axis=0)
    mse_ = np.mean(np.square(np.subtract(y_actual, y_predicted)), axis=0)
    # rmse_ = math.sqrt(mse_)
    return mse_, mean_abs_dif_

# LOAD MODEL
model = LSTM()
run.watch(model)
model.load_state_dict(torch.load("LPC_RNN_0507.pt"))
model.to(device)
#save model inputs and hyperparameters
config = run.config
config.batch_size = 20


# LOAD TEST SET
dir = "/Users/olgaseleznova/Work/Speech_recognition/DeepFormants/data/Outputs/Tracker-0507"
Xtest_raw = np.load(os.path.join(dir, "LPC_RNN_X_test.npy"))#[:, 1:].astype(np.float32).reshape(-1, 350)
Ytest_raw = np.load(os.path.join(dir, "LPC_RNN_Y_test.npy"))#[:, 1:].astype(np.float32).reshape(-1, 4)
# print(Ytest_raw.shape, Ytest_raw)


phone_collection = dict()
phone_collection["vowels"] = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay', 'eh', 'er', 'ey', 'ih', 'ix', 'iy', 'ow',
                              'oy', 'uh', 'uw', 'ux', 'axh']
phone_collection["semivowels"] = ["l", "r", "er", "w", "y", "el", "hh", "hv"]
phone_collection["nasal"] = ["em", "en", "eng", "m", "n", "ng", "nx"]
phone_collection["fricatives"] = ["s", "z", "sh", "zh", "f", "th", "v", "dh", "em", "en", "eng", "m", "n", "ng", "nx"]
phone_collection["affricates"] = ["ch", "jh"]
phone_collection["stops"] = ['b', "d", "g", "p", "t", "k", "dx", "q"]

home_dir = '/Users/olgaseleznova/Work/Speech_recognition/DeepFormants/data'
file_dir = home_dir + '/Test/'
phone_collect = defaultdict(list)
for (root, dirs, files) in os.walk(file_dir, topdown=True):
    if len(dirs) == 0:
        for file in files:
            if file.endswith(".16bit.wav"):
                # print(root, file)
                # get indices of the relevant file
                Ycurr, Ypred = get_Y_by_file(file)

                # get labels
                label_f_name = os.path.join(root, f"{file.split('.')[0]}.label")
                label_formants = file_to_array(label_f_name)
                # get vowels error per file
                for phone in list(phone_collection.keys()):
                    vow_y_actual, vow_y_predicted = get_y_by_phone(label_formants, Ycurr, Ypred, phone_collection.get(phone))
                    phone_collect[f"{phone}_actual"].append(vow_y_actual)
                    phone_collect[f"{phone}_predicted"].append(vow_y_predicted)

for phone in list(phone_collection.keys()):
    mse, mean_abs_dif = get_error(np.hstack(phone_collect[f"{phone}_actual"]), np.hstack(phone_collect[f"{phone}_predicted"]))
    print(f"mean absolute difference {phone} is {mean_abs_dif*1000}")

