import os
import sys
import numpy as np
import csv

# Add the parent of the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from extract_features import build_data, build_single_feature_row


def load_labels(label_file):
    """
    Load label file and parse formant information for every frame (10ms).

    Args:
        label_file (str): Path to the label file.

    Returns:
        list: A list of formants for every frame.
    """
    formants_list = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            formants = [float(f)/1000.0 for f in row[1:]]
            formants_list.append(formants)
    return formants_list


def create_features(input_wav_filename):
    """
    Extract features for every 10ms frame in the WAV file.

    Args:
        input_wav_filename (str): Path to the WAV file.

    Returns:
        list: A list of feature vectors, one per 10ms frame.
    """
    X = build_data(input_wav_filename)  # Keep as a list
    if len(X) == 0:
        print(f"File {input_wav_filename} is too short.")
        return None

    features = []
    for frame in X:
        # Ensure the frame has valid data before extracting features
        if len(frame) >= 17:
            features.append(build_single_feature_row(frame))
        else:
            print(f"Frame too short in file: {input_wav_filename}")
    return features


def process_directory(input_dir, output_file):
    """
    Process all files in a directory and save results as a NumPy file.

    Args:
        input_dir (str): Path to the directory containing label and WAV files.
        output_file (str): Path to save the resulting NumPy file.
    """
    data = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_file = os.path.join(root, file)
                label_file = os.path.splitext(wav_file)[0] + ".label"

                if not os.path.exists(label_file):
                    print(f"Label file missing for {wav_file}")
                    continue

                print(f"Processing: {wav_file}")
                formants = load_labels(label_file)
                features = create_features(wav_file)

                if features is None:
                    continue

                # Combine formants and features for each frame
                for frame_index, (frame_formants, frame_features) in enumerate(zip(formants, features)):
                    entry_name = f"{'_'.join(wav_file.replace('.wav', '').split('/')[-3:])}_frame_{frame_index}"
                    row = [entry_name] + frame_formants + frame_features
                    data.append(row)

    # Save the processed data
    np_data = np.array(data, dtype=object)
    np.save(output_file, np_data)
    print(f"Data saved to {output_file}")


def main():
    train_input_dir = "/home/datasets/public/formants/vtr/shua_processed/Train/"
    test_input_dir = "/home/datasets/public/formants/vtr/shua_processed/Test/"
    output_dir = "/home/datasets/public/formants/vtr/shua_processed/Outputs/"
    os.makedirs(output_dir, exist_ok=True)

    train_output_file = os.path.join(output_dir, "Train_tracker.npy")
    test_output_file = os.path.join(output_dir, "Test_tracker.npy")

    print("Processing Train directory...")
    process_directory(train_input_dir, train_output_file)

    print("Processing Test directory...")
    process_directory(test_input_dir, test_output_file)


if __name__ == "__main__":
    main()
