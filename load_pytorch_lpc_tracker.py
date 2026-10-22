import torch
import numpy as np
import csv
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "pytorchFormants"))

from weights import resolve

DEFAULT_MODEL = None  # resolved via pytorchFormants.weights


# Same architecture as pytorchFormants/Tracker/LPC_tracker.py
class LSTM(torch.nn.Module):
    def __init__(self, input_dim=350):
        super(LSTM, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=input_dim, hidden_size=512, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.fc = torch.nn.Linear(256, 4)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x)


def load_model(model_path, device):
    model_path = resolve('rnn', model_path)
    model = LSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def read_features(features_file):
    """Read the feature CSV written by extract_features.create_features.

    Each row is [name, f_1 ... f_350]. Returns (names, features array).
    """
    names, rows = [], []
    with open(features_file, 'r') as f:
        for row in csv.reader(f):
            if not row or len(row) < 351:
                print(f"Skipping invalid row: {row[:1]}")
                continue
            names.append(row[0])
            rows.append(row[1:351])
    if not rows:
        raise ValueError(f"No valid feature rows found in {features_file}")
    features = np.array(rows, dtype=np.float32)
    # The feature extractor can emit NaN for very short/silent frames.
    return names, np.nan_to_num(features, nan=0.0)


def predict_from_features(features_file, preds_file, model_path=DEFAULT_MODEL):
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found at {features_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    names, features = read_features(features_file)

    # The whole file is one sequence, so the LSTM keeps context across frames.
    inputs = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(inputs).squeeze(0).cpu().numpy()

    with open(preds_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['NAME', 'F1', 'F2', 'F3', 'F4'])
        for name, pred in zip(names, preds):
            writer.writerow([name] + (1000 * pred).tolist())


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python load_pytorch_lpc_tracker.py <features_file> <predictions_file> [model_path]")
        sys.exit(1)

    predict_from_features(sys.argv[1], sys.argv[2],
                          sys.argv[3] if len(sys.argv) == 4 else DEFAULT_MODEL)
