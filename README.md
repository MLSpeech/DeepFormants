DeepFormants
============

Shua Dissen (shua.dissen@gmail.com)

Joseph Keshet (joseph.keshet@biu.ac.il)             

DeepFormants is a software package for formant tracking and estimation, using two algorithms based on deep networks. It works as follows:
* The user provides a wav file with an initial stop consonant. 
* Optionally, the user can designate a window for estimation by providing a start and end time (specified in seconds).
* A classifer is used to estimate the formants in the file, with two modes:
* Estimation: If a time window is specified, a single estimate is made for F1-F3 within that window. 
* Tracking: If no time window is given, the model will track F1-F3 and give a measurement at every 10 milliseconds across the length of the file.

This is a beta version of DeepFormants. Any reports of bugs, comments on how to improve the software or documentation, or questions are greatly appreciated, and should be sent to the authors at the addresses given above.

How to use:

For vowel formant estimation, call the main script in a terminal with the following inputs: wav file, formant output filename, and the vowel begin and end times:

```Ex: $ python formants.py Example.wav ExamplePredictions.csv --begin 1.2 --end 1.3```

For formant tracking, just call the script with the wav file and output filename:

```Ex: $ python formants.py Example.wav ExamplePredictions.csv```

Installation instructions

Download the code.

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

recommended use the pre-trained estimation and tracking models.

The Estimation model can be downloaded here and because of size constraints the Tracking model can be abtained by download from this link

http://u.cs.biu.ac.il/~jkeshet/deep_formants/Tracking_model.dat


TODO

add training code
