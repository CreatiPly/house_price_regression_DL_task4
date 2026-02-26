import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle

housing = fetch_california_housing()

print(f"Feature Names: {housing.feature_names}")
print(f"Data Shape: {housing.data.shape}")

X = housing.data
y = housing.target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
# X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# train_loader = DataLoader(
#     TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True
# )
# val_loader = DataLoader(
#     TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False
# )


# class SimpleRegressionNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linearLayer1 = nn.Linear(8, 50)
#         self.linearLayer2 = nn.Linear(50, 100)
#         self.relu = nn.ReLU()
#         self.linearLayer3 = nn.Linear(100, 1)

#     def forward(self, x):
#         u = self.linearLayer1(x)
#         v = self.relu(u)
#         w = self.linearLayer2(v)
#         m = self.relu(w)
#         return self.linearLayer3(m)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Target Device Selected: {device.upper()}")
# model = SimpleRegressionNet().to(device)

# loss_fn = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# writer = SummaryWriter("runs/housing_regression")

# epochs = 1000
# best_val_loss = float("inf")
