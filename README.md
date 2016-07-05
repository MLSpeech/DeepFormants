DeepFormants
============

Shua Dissen (shua.dissen@gmail.com)            
Joseph Keshet (joseph.keshet@biu.ac.il)  


DeepFormants is a software package for formant tracking and estimation, using two algorithms based on deep networks. It works as follows:
* The user provides a wav file with an initial stop consonant. 
* Optionally, the user can designate a window for estimation by providing a start and end time (specified in seconds).
* A classifer is used to estimate the formants in the file, with two modes:
* Estimation: If a time window is specified, a single estimate is made for F1-F4 within that window. 
* Tracking: If no time window is given, the model will track F1-F4 and give a measurement at every 10 milliseconds across the length of the file.

This is a beta version of DeepFormants. Any reports of bugs, comments on how to improve the software or documentation, or questions are greatly appreciated, and should be sent to the authors at the addresses given above.

---


## Installation instructions

Download the code. The code is based on signal processing package in Python called [Talkbox] (https://pypi.python.org/pypi/scikits.talkbox) and a deep networks package called [Torch] (torch.ch).

Dependencies:
Run these lines in a terminal to install everything necessary for feature extraction.
```
sudo apt-get install python-numpy python-scipy python-nose python-pip

sudo pip install scikits.talkbox 
```
Next for the installation of Torch for loading the models run this.
```
git clone https://github.com/torch/distro.git ~/torch --recursive

cd ~/torch; bash install-deps;

./install.sh
```
```
luarocks install rnn
```
The Estimation model can be downloaded here and because of size constraints the Tracking model can be abtained by download from this link
[tracking_model.mat] (https://drive.google.com/open?id=0Bxkc5_D0JjpiZWx4eTU1d0hsVXc)

## How to use:

For vowel formant estimation, call the main script in a terminal with the following inputs: wav file, formant output filename, and the vowel begin and end times:

```
python formants.py data/Example.wav data/ExamplePredictions.csv --begin 1.2 --end 1.3
```

or the vowel begin and end times can be taken from a TextGrid file (here the name of the TextGrid is Example.TextGrid and the vowel is taken from a tier called "VOWEL"):

```
python formants.py data/Example.wav data/examplePredictions.csv --textgrid_filename data/Example.TextGrid \
          --textgrid_tier VOWEL
```

For formant tracking, just call the script with the wav file and output filename:

```
python formants.py data/Example.wav data/ExamplePredictions.csv
```


## TODO

Add training code.


