import os
import sys
import numpy as np
import csv

# Add the parent of the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from extract_features import build_data, build_single_feature_row


def load_labels(label_file):
    """
    Load label file and parse formant information for vowel segments.

    Each row in the label file contains: phoneme, F1, F2, F3, F4.
    Time is determined by the row number * 10ms.

    Args:
        label_file (str): Path to the label file.

    Returns:
        list: A list of tuples (vowel, start_time, end_time, [formants]).
    """
    vowels = {
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay',
        'eh', 'er', 'ey', 'ih', 'ix', 'iy', 'ow', 'oy',
        'uh', 'uw', 'ux'
    }
    segments = []
    current_vowel = None
    formants = []
    start_time = None

    with open(label_file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for row_index, row in enumerate(reader):
            label = row[0]
            current_formants = [float(f) for f in row[1:]]
            time_stamp = row_index * 0.01  # Time in seconds (row number * 10ms)

            if label in vowels:
                if current_vowel is None:
                    # Start a new vowel segment
                    current_vowel = label
                    start_time = time_stamp
                elif label != current_vowel:
                    # Vowel switches; finalize the previous vowel
                    end_time = time_stamp
                    avg_formants = (np.mean(formants, axis=0) / 1000).tolist()
                    segments.append((current_vowel, start_time, end_time, avg_formants))
                    # Start the new vowel segment
                    current_vowel = label
                    start_time = time_stamp
                    formants = []
                formants.append(current_formants)
            else:
                # Non-vowel row encountered; finalize the current vowel segment
                if current_vowel is not None:
                    end_time = time_stamp
                    avg_formants = (np.mean(formants, axis=0) / 1000).tolist()
                    segments.append((current_vowel, start_time, end_time, avg_formants))
                    current_vowel = None
                    formants = []

    # Finalize any remaining vowel at the end of the file
    if current_vowel is not None:
        end_time = (row_index + 1) * 0.01  # End of the last row
        avg_formants = (np.mean(formants, axis=0) / 1000).tolist()
        segments.append((current_vowel, start_time, end_time, avg_formants))

    return segments



def extract_features_for_segment(wav_file, start, end):
    """
    Extract features for a specific segment of a WAV file.
    
    Args:
        wav_file (str): Path to the WAV file.
        start (float): Start time in seconds.
        end (float): End time in seconds.

    Returns:
        list: Extracted features or None if data is too small.
    """
    feature_data = build_data(wav_file, start, end)
    if feature_data.size < 17:
        return None
    return build_single_feature_row(feature_data)


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
                segments = load_labels(label_file)

                for i, (vowel, start, end, formants) in enumerate(segments):
                    features = extract_features_for_segment(wav_file, start, end)
                    if features is not None:
                        entry_name = f"{'_'.join(wav_file.replace('.wav','').split('/')[-3:])}_{vowel}_{i}"
                        row = [entry_name] + formants + features
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

    train_output_file = os.path.join(output_dir, "Train.npy")
    test_output_file = os.path.join(output_dir, "Test.npy")

    print("Processing Train directory...")
    process_directory(train_input_dir, train_output_file)

    print("Processing Test directory...")
    process_directory(test_input_dir, test_output_file)


if __name__ == "__main__":
    main()
