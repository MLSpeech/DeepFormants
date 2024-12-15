import torch
import numpy as np
import csv
import sys
import os

# Define the PyTorch model
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

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

model_path = "pytorchFormants/Estimator/LPC_NN_scaledLoss.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Main prediction logic
def predict_from_features(features_file, preds_file):
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found at {features_file}")

    with open(features_file, 'r') as f, open(preds_file, 'w', newline='') as out_f:
        reader = csv.reader(f)
        writer = csv.writer(out_f)

        # Write header
        writer.writerow(['NAME', 'F1', 'F2', 'F3', 'F4'])

        for row in reader:
            if not row or len(row) < 351:  # Check for valid row
                print(f"Skipping invalid row: {row}")
                continue

            # Extract the name and features
            name = row[0]
            features = np.array(row[1:], dtype=np.float32).reshape(1, -1)

            # Convert features to a PyTorch tensor
            features_tensor = torch.from_numpy(features).to(device)

            # Perform prediction
            with torch.no_grad():
                prediction = model(features_tensor).cpu().numpy()

            # Scale predictions by 1000
            scaled_prediction = 1000 * prediction[0]

            # Write results to the output file
            writer.writerow([name] + scaled_prediction.tolist())

# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python model_predict.py <features_file> <predictions_file>")
        sys.exit(1)

    features_file = sys.argv[1]
    preds_file = sys.argv[2]

    predict_from_features(features_file, preds_file)
