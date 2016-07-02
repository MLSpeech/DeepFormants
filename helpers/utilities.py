# Copyright (c) 2014 Joseph Keshet, Morgan Sonderegger, Thea Knowles
#
# This file is part of Autovot, a package for automatic extraction of
# voice onset time (VOT) from audio files.
#
# Autovot is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Autovot is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Autovot.  If not, see
# <http://www.gnu.org/licenses/>.
#

import subprocess
import random
import logging
import wave
import tempfile
import os


def csv_append_row(tmp_preds, preds_filename, with_headers=True):

    if with_headers:
        skip_header = True

    all_lines = list()

    # check if the CSV file exists
    if os.path.isfile(preds_filename):
        # read it lines
        for line in open(preds_filename, 'r'):
            all_lines.append(line)
    else:
        # if the file does not exist it does not have headers and they should be copied
        skip_header = False

    # check if there is a header
    for line in open(tmp_preds, 'r'):
        if skip_header:
            skip_header = False
        else:
            all_lines.append(line)
    # now dump everything back
    with open(preds_filename, 'w') as f:
        for line in all_lines:
            f.write(line)


def generate_tmp_filename(extension):
    return tempfile._get_default_tempdir() + "/" + next(tempfile._get_candidate_names()) + "." + extension


def logging_defaults(logging_level="INFO"):
    logging.basicConfig(level=logging_level, format='%(asctime)s.%(msecs)d [%(filename)s] %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')

def num_lines(filename):
    lines = 0
    for _ in open(filename, 'rU'):
        lines += 1
    return lines


def easy_call(command):
    try:
        logging.debug(command)
        return_code = subprocess.call(command, shell=True)
        if return_code == 127 or return_code < 0:
            logging.debug('Return code: %d' % return_code)
            exit(-1)
    except Exception as exception:
        logging.error('Could not execute the following:')
        logging.error(command)
        logging.error('%s - %s' % (type(exception), exception.args))
        exit(-1)


def random_shuffle_data(in_features_filename, in_labels_filename, out_features_filename, out_labels_filename):

    # open files
    in_features = open(in_features_filename, 'rU')
    in_labels = open(in_labels_filename, 'rU')

    # read infra text header
    header = in_labels.readline()
    dims = header.split()

    # read file lines
    lines = list()
    for x, y in zip(in_features, in_labels):
        lines.append((x, y))
    if len(lines) != int(dims[0]):
        logging.error("Either the feature file and the label file are not the same length of label file missing a "
                      "header")
        exit(-1)

    # close files
    in_features.close()
    in_labels.close()

    # random shuffle the instances
    random.shuffle(lines)

    # write back the result
    out_features = open(out_features_filename, 'w')
    out_labels = open(out_labels_filename, 'w')

    # write labels header
    header = "%s %s\n" % (dims[0], dims[1])
    out_labels.write(header)

    # write data
    for x, y in lines:
        out_features.write(x)
        out_labels.write(y)

    # close files
    out_features.close()
    out_labels.close()

    return len(lines)


def extract_lines(input_filename, output_filename, lines_range, has_header=False):

    if lines_range[0] >= lines_range[1]:
        logging.error("Range should be causal.")
        exit(-1)
    input_file = open(input_filename, 'rU')
    output_file = open(output_filename, 'w')
    if has_header:
        header = input_file.readline().strip().split()
        new_header = "%d 2\n" % (lines_range[1]-lines_range[0]+1)
        output_file.write(new_header)
    for line_num, line in enumerate(input_file):
        if lines_range[0] <= line_num <= lines_range[1]:
            output_file.write(line)
    input_file.close()
    output_file.close()


def is_textgrid(filename):
    try:
        file = open(filename, 'rU')
        first_line = file.readline()
    except:
        return False
    if "ooTextFile" in first_line:
        return True
    return False


def is_valid_wav(filename):
    # check the sampling rate and number bits of the WAV
    try:
        wav_file = wave.Wave_read(filename)
    except:
        return False
    if wav_file.getframerate() != 16000 or wav_file.getsampwidth() != 2 or wav_file.getnchannels() != 1 \
        or wav_file.getcomptype() != 'NONE':
        return False
    return True
