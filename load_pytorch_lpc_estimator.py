import torch
import numpy as np
import csv
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "pytorchFormants"))

from weights import resolve

DEFAULT_MODEL = None  # resolved via pytorchFormants.weights


# Same architecture as pytorchFormants/Estimator/LPC_Estimate_Class.py
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Dense1 = torch.nn.Linear(350, 1024)
        self.Dense2 = torch.nn.Linear(1024, 512)
        self.Dense3 = torch.nn.Linear(512, 256)
        self.out = torch.nn.Linear(256, 4)

    def forward(self, x):
        x = torch.sigmoid(self.Dense1(x))
        x = torch.sigmoid(self.Dense2(x))
        x = torch.sigmoid(self.Dense3(x))
        return self.out(x)


def load_model(model_path, device):
    model_path = resolve('ff', model_path)
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_from_features(features_file, preds_file, model_path=DEFAULT_MODEL):
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found at {features_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    with open(features_file, 'r') as f, open(preds_file, 'w', newline='') as out_f:
        reader = csv.reader(f)
        writer = csv.writer(out_f)
        writer.writerow(['NAME', 'F1', 'F2', 'F3', 'F4'])

        for row in reader:
            if not row or len(row) < 351:
                print(f"Skipping invalid row: {row[:1]}")
                continue

            name = row[0]
            feats = np.array(row[1:351], dtype=np.float32).reshape(1, -1)
            feats = np.nan_to_num(feats, nan=0.0)

            with torch.no_grad():
                prediction = model(torch.from_numpy(feats).to(device)).cpu().numpy()

            writer.writerow([name] + (1000 * prediction[0]).tolist())


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python load_pytorch_lpc_estimator.py <features_file> <predictions_file> [model_path]")
        sys.exit(1)

    predict_from_features(sys.argv[1], sys.argv[2],
                          sys.argv[3] if len(sys.argv) == 4 else DEFAULT_MODEL)
