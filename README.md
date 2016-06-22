# DeepFormants
Formant Tracking &amp; Estimation

How to use:

For vowel formant estimation call the main script in a terminal with the wav file, formant output filename and the vowel begin and end times

```Ex: $ python formants.py Example.wav examplePredictions --begin 1.2 --end 1.3```

For formant tracking just call the script with the wav file and output filename and it will return the formants for the whole wav file every 10 ms

```Ex: $ python formants.py Example.wav examplePredictions```

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

Reccomnended use the pre-trained estimation and tracking models.

The Estimation model can be downloaded here and because of size constraints the Tracking model can be abtained by download from ths link

https://Link/to/Tracking/model

Yossi ?
