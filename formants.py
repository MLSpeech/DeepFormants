import Extract_Features as features
from subprocess import call
import os
import sys
import shlex
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
wav_file = sys.argv[1]
if len(sys.argv) == 4:
	begin = sys.argv[2]
	end = sys.argv[3]
	Data = features.Create_features(str(wav_file),float(begin),float(end))
	ff = str(os.path.dirname(os.path.realpath(__file__))+'/Features/features_' + wav_file.replace('.wav','.txt'))
	easy_call("th load_estimation_model.lua " + ff)

elif len(sys.argv) == 2:
	Data = features.Create_features(str(wav_file))
	ff = str(os.path.dirname(os.path.realpath(__file__))+'/Features/features_' + wav_file.replace('.wav','.txt'))
	easy_call("th load_tracking_model.lua " + ff)
else:
	print('wrong amount of parameters')



