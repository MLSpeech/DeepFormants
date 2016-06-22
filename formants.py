import Extract_Features as features
from subprocess import call
import os
import sys
import shlex
import argparse


def easy_call(command, debug_mode=False):
    try:
        #command = "time " + command
        if debug_mode:
            print >>sys.stderr, command
        call(command, shell=True)
    except Exception as exception:
        print "Error: could not execute the following"
        print ">>", command
        print type(exception)     # the exception instance
        print exception.args      # arguments stored in .args
        exit(-1)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Extract features for formants estimation.')
    parser.add_argument('wav_file', default='', help="WAV audio filename (single vowel or an whole utternace)")
    parser.add_argument('formants_file', default='', help="output formant text file")
    parser.add_argument('--begin', help="beginning time in the WAV file", default=0.0, type=float)
    parser.add_argument('--end', help="end time in the WAV file", default=-1.0, type=float)
    args = parser.parse_args()
    full_path = os.path.realpath(__file__)
    if not os.path.exists(os.path.dirname(full_path)+'/Features/'):
        os.makedirs(os.path.dirname(full_path)+'/Features/')
    if not os.path.exists(os.path.dirname(full_path)+'/Predictions/'):
        os.makedirs(os.path.dirname(full_path)+'/Predictions/')

    if args.begin > 0.0 or args.end > 0.0:
    	Data = features.Create_features(args.wav_file, args.formants_file, args.begin, args.end)
    	ff = str(os.path.dirname(os.path.realpath(__file__))+'/Features/features_' + args.formants_file+'.txt')
	pf = str(os.path.dirname(os.path.realpath(__file__))+'/Predictions/' +args.formants_file+'.csv')
	easy_call("th load_estimation_model.lua " + ff + ' ' + pf)
    else:
    	Data = features.Create_features(args.wav_file, args.formants_file)
    	ff = str(os.path.dirname(os.path.realpath(__file__))+'/Features/features_' + args.formants_file+'.txt')
	pf = str(os.path.dirname(os.path.realpath(__file__))+'/Predictions/' +args.formants_file+'.csv')
	easy_call("th load_tracking_model.lua " + ff + ' ' + pf)

