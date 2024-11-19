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
        arcep_mat = []
        for i in range(len(X)):
            arr = []
            arr.extend(build_single_feature_row(X[i], Atal))
            arcep_mat.append(arr)
        return arcep_mat
    else:
        print(f"Data {input_wav_filename, begin, end} is too small")
        return None


def create_labels(labels, begin: float, end: float):
    vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay', 'eh', 'er', 'ey', 'ih', 'ix', 'iy', 'ow', 'oy', 'uh', 'uw', 'ux']
    begin_, end_ = round(begin), round(end)
    labels_list = labels[begin_: end_, 0]
    formants_list = labels[begin_: end_, 1:]
    labels_collect = dict()
    indices = list()
    for ind, lab in enumerate(labels_list):
        if lab in vowels: #and lab in labels_collect.keys():
            indices.append(ind)
            if lab != labels_list[ind+1] or labels_list.size == ind:
                formts_gr = formants_list[indices].astype(float)
                aver = np.mean(formts_gr, axis=0)
                labels_collect[(lab, indices[0], indices[-1])] = np.divide(aver, 1000).tolist()
                indices = list()
            else:
                continue
        else:
            continue
    return labels_collect


home_dir = '/Users/olgaseleznova/Work/Speech_recognition/DeepFormants/data'
file_dir = home_dir + '/Train/'
output_list = list()
for (root, dirs, files) in os.walk(file_dir, topdown=True):
    if len(dirs) == 0:
        for file in files:
            if file.endswith(".16bit.wav") and file == "si1027.16bit.wav":
                wave_f_name = os.path.join(root, file)
                label_f_name = os.path.join(root, f"{file.split('.')[0]}.label")
                print(root, file)
                est_begin = 0.0
                est_end = -1
                label_formants = file_to_array(os.path.join(root, label_f_name))
                labels = create_labels(label_formants, est_begin, est_end)
                for key in labels.keys():
                    features = create_features(wave_f_name, key[1] / 100.0, (key[2]) / 100.0)
                    if features is not None:
                        features_formants = np.array(['_'.join(root.split('/')[-2:])] + (labels.get(key) + features))
                        output_list.append(features_formants)
                    else:
                        continue
                print('--------------------------------')

output_arr = np.array(output_list)
print(output_arr)
output_dir = home_dir + "/Outputs/Estimator/"
np.save(os.path.join(output_dir, f"{'_'.join(file_dir.split('/')[-2:])}_output.npy"), output_arr)

et = time.time()
print('Execution time:', et - st, 'seconds')

d = np.load(os.path.join(output_dir, f"{'_'.join(file_dir.split('/')[-2:])}_output.npy"))
print(d.shape)

