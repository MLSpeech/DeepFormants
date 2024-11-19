#!/bin/bash
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

for D in $(find . -mindepth 1 -maxdepth 1 -type d); do
  echo "${D}"
  myabsfile=$(get_abs_filename ${D})
  echo $myabsfile
  cd $myabsfile
  for file in *.wav; do sox $file -r 16000 -c 1 -b 16 "$(basename $file .wav).16bit.wav" -V; done
  cd ..
done