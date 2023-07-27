import numpy as np
import csv
import itertools
from collections import defaultdict
import time
import argparse

# HELPERS
from helpers.utilities import *
from extract_features import build_data, build_single_feature_row
# get the start time
st = time.time()


def file_to_array(f_name):
    with open(f_name, 'r') as x:
        data = list(csv.reader(x, delimiter="\t"))
    return np.array(data)


def create_features(input_wav_filename, begin=None, end=None, Atal=False):
    X = build_data(input_wav_filename, begin, end)
    if X.size >= 17:
        if begin is not None and end is not None:
            arr = []
            arr.extend(build_single_feature_row(X, Atal))
            return arr
        else:
            print("Enter begin and end time of tracking formants.")
        arcep_mat = []
        for i in range(len(X)):
            arr = []
            arr.extend(build_single_feature_row(X[i], Atal))
            arcep_mat.append(arr)
        return arcep_mat
    else:
        print(f"Phoneme {input_wav_filename, begin, end} is too short")
        return None


begin = 0
# end = 700

home_dir = '/Users/olgaseleznova/Work/Speech_recognition/DeepFormants/data'
file_dir = home_dir + '/Test/'
feature_output = list()
formants = np.array([])
for (root, dirs, files) in os.walk(file_dir, topdown=True):
    if len(dirs) == 0:
        for file in files[:2]:
            if file.endswith(".16bit.wav"):
                wave_f_name = os.path.join(root, file)
                label_f_name = os.path.join(root, f"{file.split('.')[0]}.label")
                # output_f = '_'.join(root.split('/')[-2:])
                print(root, file)
                label_formants = file_to_array(label_f_name)
                end = label_formants.shape[0]+1
                labels_filt = label_formants[begin:end, 1:]
                labels_div = np.divide(labels_filt.astype(float), 1000)
                # np.divide(aver, 1000).tolist()
                formants_arr = np.column_stack(([file]*labels_div.shape[0], labels_div))
                if formants.size == 0:
                    formants = formants_arr
                else:
                    formants = np.row_stack((formants, formants_arr))
                # formants = np.append(formants, formants_arr)
                # formants.append(formants_arr)
                # row_num = begin

                for row_num in range(labels_filt.shape[0]):
                    features = create_features(wave_f_name, row_num/100.0, (row_num+1)/100.0)
                    if features is not None:
                        features_formants = np.array([file] + features)
                        feature_output.append(features_formants)
                        # row_num += 1
            else:
                continue

x_train_fname, y_train_fname = "LPC_RNN_X_test.npy", 'LPC_RNN_Y_test.npy'

features_output = np.array(feature_output)
formants_array = np.array(formants)
print(features_output.shape, features_output)
print(formants_array.shape, formants_array)
output_dir = home_dir + "/Outputs/Tracker-0507/"
np.save(os.path.join(output_dir, x_train_fname), features_output)
np.save(os.path.join(output_dir, y_train_fname), formants_array)

et = time.time()
print('Execution time:', et - st, 'seconds')
