
import extract_features as features
import argparse
from helpers.textgrid import *
from helpers.utilities import *
import shutil

def predict_from_times(wav_filename, preds_filename, begin, end):
    tmp_features_filename = tempfile._get_default_tempdir() + "/" + next(tempfile._get_candidate_names()) + ".txt"
    print tmp_features_filename

    if begin > 0.0 or end > 0.0:
        features.create_features(wav_filename, tmp_features_filename, begin, end)
        easy_call("th load_estimation_model.lua " + tmp_features_filename + ' ' + preds_filename)
    else:
        features.create_features(wav_filename, tmp_features_filename)
        easy_call("th load_tracking_model.lua " + tmp_features_filename + ' ' + preds_filename)


def predict_from_textgrid(wav_filename, preds_filename, textgrid_filename, textgrid_tier):

    print wav_filename

    if os.path.exists(preds_filename):
        os.remove(preds_filename)

    textgrid = TextGrid()

    # read TextGrid
    textgrid.read(textgrid_filename)

    # extract tier names
    tier_names = textgrid.tierNames()

    if textgrid_tier in tier_names:
        tier_index = tier_names.index(textgrid_tier)
        # run over all intervals in the tier
        for interval in textgrid[tier_index]:
            if re.search(r'\S', interval.mark()):
                tmp_features_filename = generate_tmp_filename()
                tmp_preds = generate_tmp_filename()
                features.create_features(wav_filename, tmp_features_filename, interval.xmin(), interval.xmax())
                easy_call("th load_estimation_model.lua " + tmp_features_filename + ' ' + tmp_preds)
                csv_append_row(tmp_preds, preds_filename)
    else:  # process first tier
        for interval in textgrid[0]:
            if re.search(r'\S', interval.mark()):
                tmp_features_filename = generate_tmp_filename()
                tmp_preds = generate_tmp_filename()
                features.create_features(wav_filename, tmp_features_filename, interval.xmin(), interval.xmax())
                easy_call("th load_estimation_model.lua " + tmp_features_filename + ' ' + tmp_preds)
                csv_append_row(tmp_preds, preds_filename)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Estimation and tracking of formants.')
    parser.add_argument('wav_file', default='', help="WAV audio filename (single vowel or an whole utternace)")
    parser.add_argument('formants_file', default='', help="output formant CSV file")
    parser.add_argument('--textgrid_filename', default='', help="get beginning and end times from a TextGrid file")
    parser.add_argument('--textgrid_tier', default='', help="a tier name with portion to process (default first tier)")
    parser.add_argument('--begin', help="beginning time in the WAV file", default=0.0, type=float)
    parser.add_argument('--end', help="end time in the WAV file", default=-1.0, type=float)
    args = parser.parse_args()

    if args.textgrid_filename:
        predict_from_textgrid(args.wav_file, args.formants_file, args.textgrid_filename, args.textgrid_tier)
    else:
        predict_from_times(args.wav_file, args.formants_file, args.begin, args.end)

