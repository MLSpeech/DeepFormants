
import extract_features as features
import argparse
import numpy as np
from helpers.textgrid import *
from helpers.utilities import *
import shutil

from load_pytorch_lpc_estimator import predict_from_features as estimate_from_features
from load_pytorch_lpc_tracker import predict_from_features as track_from_features


def track(wav_filename, preds_filename, backend="rnn", model_path=None):
    """Track F1-F4 every 10ms across the whole file."""
    if backend == "crnn":
        # The CRNN reads the waveform directly, it does not use the LPC feature CSV.
        from pytorchFormants.CRNN import crnn_inference
        crnn_inference.predict_from_wav(wav_filename, preds_filename,
                                        model_path or crnn_inference.DEFAULT_MODEL)
        return

    tmp_features_filename = generate_tmp_filename("features")
    features.create_features(wav_filename, tmp_features_filename)
    if model_path:
        track_from_features(tmp_features_filename, preds_filename, model_path)
    else:
        track_from_features(tmp_features_filename, preds_filename)


def estimate(wav_filename, preds_filename, begin, end, backend="ff", model_path=None):
    """Estimate a single F1-F4 for the [begin, end] window."""
    if backend == "cnn":
        # The CNN reads the waveform directly and builds a 55x50 spectrogram image.
        from pytorchFormants.Estimator import cnn_inference
        cnn_inference.predict_from_wav(wav_filename, preds_filename, begin, end,
                                       model_path or cnn_inference.DEFAULT_MODEL)
        return

    tmp_features_filename = generate_tmp_filename("features")
    features.create_features(wav_filename, tmp_features_filename, begin, end)
    if model_path:
        estimate_from_features(tmp_features_filename, preds_filename, model_path)
    else:
        estimate_from_features(tmp_features_filename, preds_filename)


def predict_from_times(wav_filename, preds_filename, begin, end, backend=None, model_path=None):

    if begin > 0.0 or end > 0.0:
        estimate(wav_filename, preds_filename, begin, end, backend or "ff", model_path)
    else:
        track(wav_filename, preds_filename, backend or "rnn", model_path)


def predict_from_textgrid(wav_filename, preds_filename, textgrid_filename, textgrid_tier,
                          model_path=None, backend="ff"):

    print(wav_filename)

    if os.path.exists(preds_filename):
        os.remove(preds_filename)

    textgrid = TextGrid()

    # read TextGrid
    textgrid.read(textgrid_filename)

    # extract tier names
    tier_names = textgrid.tierNames()

    if textgrid_tier in tier_names:
        tier = textgrid[tier_names.index(textgrid_tier)]
    else:  # process first tier
        tier = textgrid[0]

    intervals = [i for i in tier if re.search(r'\S', i.mark())]
    if not intervals:
        print("No non-empty intervals found, nothing to predict.")
        return

    if backend == "cnn":
        from pytorchFormants.Estimator import cnn_inference
        import csv as _csv
        device_model = None
        with open(preds_filename, 'w', newline='') as out_f:
            writer = _csv.writer(out_f)
            writer.writerow(['NAME', 'F1', 'F2', 'F3', 'F4'])
            for interval in intervals:
                tmp_preds = generate_tmp_filename("csv")
                pred = cnn_inference.predict_from_wav(
                    wav_filename, tmp_preds, interval.xmin(), interval.xmax(),
                    model_path or cnn_inference.DEFAULT_MODEL)
                writer.writerow([wav_filename] + (1000 * pred).tolist())
                os.remove(tmp_preds)
        return

    # Collect every interval into a single feature file so the model is loaded once.
    rows = []
    for interval in intervals:
        tmp_features_filename = generate_tmp_filename("features")
        rows.append(features.create_features(wav_filename, tmp_features_filename,
                                             interval.xmin(), interval.xmax()))

    tmp_all_features = generate_tmp_filename("features")
    np.savetxt(tmp_all_features, np.asarray(rows), delimiter=",", fmt="%s")
    if model_path:
        estimate_from_features(tmp_all_features, preds_filename, model_path)
    else:
        estimate_from_features(tmp_all_features, preds_filename)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Estimation and tracking of formants.')
    parser.add_argument('wav_file', default='', help="WAV audio filename (single vowel or an whole utternace)")
    parser.add_argument('formants_file', default='', help="output formant CSV file")
    parser.add_argument('--textgrid_filename', default='', help="get beginning and end times from a TextGrid file")
    parser.add_argument('--textgrid_tier', default='', help="a tier name with portion to process (default first tier)")
    parser.add_argument('--begin', help="beginning time in the WAV file", default=0.0, type=float)
    parser.add_argument('--end', help="end time in the WAV file", default=-1.0, type=float)
    parser.add_argument('--backend', default=None, choices=['ff', 'cnn', 'rnn', 'crnn'],
                        help="which model to use. Estimation: 'ff' (dense net over LPC features, default) "
                             "or 'cnn' (spectrogram image). Tracking: 'rnn' (LSTM over LPC features, "
                             "default) or 'crnn' (ConvLSTM over the spectrogram).")
    parser.add_argument('--model', default='',
                        help="override the checkpoint for the chosen backend")
    args = parser.parse_args()

    tracking = not args.textgrid_filename and not (args.begin > 0.0 or args.end > 0.0)
    if args.backend in (('ff', 'cnn') if tracking else ('rnn', 'crnn')):
        parser.error(f"--backend {args.backend} is "
                     f"{'an estimation' if tracking else 'a tracking'} model, but this call is "
                     f"{'tracking' if tracking else 'estimation'}. "
                     f"Use {'rnn/crnn' if tracking else 'ff/cnn'}.")

    if args.textgrid_filename:
        predict_from_textgrid(args.wav_file, args.formants_file, args.textgrid_filename,
                              args.textgrid_tier, args.model, args.backend or 'ff')
    else:
        predict_from_times(args.wav_file, args.formants_file, args.begin, args.end,
                           args.backend, args.model)
