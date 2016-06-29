#!/bin/bash

if [ $# -eq 2 ]
then
    tempfile=`mktemp -t txt`
    python extract_features.py $1 $tempfile
    th load_estimation_model.lua $tempfile $2
elif [ $# -eq 4 ]
then
    tempfile=`mktemp -t txt`
    python extract_features.py $1 $tempfile --begin $3 --end $4
    th load_estimation_model.lua $tempfile $2
else
    echo "$0 wav_filename pred_csv_filename [begin_time end_time]"
fi
