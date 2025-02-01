import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

CSV_URL = 'https://raw.githubusercontent.com/HamedTabkhi/Intro-to-ML/refs/heads/main/Dataset/Housing.csv'
df_raw = pd.read_csv(CSV_URL)

df_raw['price'] = df_raw['price'] / 1e6

def encode_yes_no(df, columns):
    for c in columns:
        df[c] = df[c].map({'yes': 1, 'no': 0})
    return df

def make_dataloader(X, y, batch_size=64, shuffle=True):
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).view(-1, 1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def evaluate_regression(model, X, y, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        preds = model(X_t)
        preds = preds.cpu().numpy()
    mse_val = mean_squared_error(y, preds)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y, preds)
    return mse_val, rmse_val, r2_val

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLPRegressorComplex(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressorComplex, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train_model(model, train_loader, X_val, y_val,
                num_epochs=50, lr=1e-3, device='cpu',
                title="MLP Training"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).view(-1,1).to(device)

    train_mse_list = []
    val_mse_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        train_mse = running_loss / total_samples
        train_mse_list.append(train_mse)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_mse_list.append(val_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train MSE: {train_mse:.4f}, Val MSE: {val_loss:.4f}")

    plt.figure()
    plt.plot(train_mse_list, label='Train MSE')
    plt.plot(val_mse_list, label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.show()

    mse_val, rmse_val, r2_val = evaluate_regression(model, X_val, y_val, device=device)
    print(f"Final Val RMSE: {rmse_val:.4f}")
    print(f"Final Val R^2: {r2_val:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print("Model complexity (Total params):", total_params)
    print("Model structure:\n", model)

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yes_no_cols = [
        'mainroad','guestroom','basement','hotwaterheating',
        'airconditioning','prefarea'
    ]

    print("\n=== 2.a ===")
    df_no_ohe = df_raw.copy()

    # 目标列 y
    y_no_ohe = df_no_ohe['price'].values  # price已除以1e6
    df_no_ohe.drop('price', axis=1, inplace=True)

    df_no_ohe = encode_yes_no(df_no_ohe, yes_no_cols)

    f_map = {'furnished':2, 'semi-furnished':1, 'unfurnished':0}
    df_no_ohe['furnishingstatus'] = df_no_ohe['furnishingstatus'].map(f_map)

    X_no_ohe = df_no_ohe.values.astype(np.float32)

    scaler_a = StandardScaler()
    X_no_ohe = scaler_a.fit_transform(X_no_ohe)

    X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(
        X_no_ohe, y_no_ohe, test_size=0.2, random_state=42
    )

    train_loader_a = make_dataloader(X_train_a, y_train_a, batch_size=64, shuffle=True)

    model_a = MLPRegressor(input_dim=X_no_ohe.shape[1])
    model_a = train_model(model_a,
                          train_loader_a,
                          X_val_a, y_val_a,
                          num_epochs=50, lr=1e-3,
                          device=device,
                          title="2.a No One-Hot")

    print("\n=== 2.b ===")
    df_ohe = df_raw.copy()

    y_ohe = df_ohe['price'].values
    df_ohe.drop('price', axis=1, inplace=True)

    df_ohe = encode_yes_no(df_ohe, yes_no_cols)

    f_ohe = pd.get_dummies(df_ohe['furnishingstatus'], prefix='fs')
    df_ohe.drop('furnishingstatus', axis=1, inplace=True)
    df_ohe = pd.concat([df_ohe, f_ohe], axis=1)

    X_ohe = df_ohe.values.astype(np.float32)

    scaler_b = StandardScaler()
    X_ohe = scaler_b.fit_transform(X_ohe)

    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
        X_ohe, y_ohe, test_size=0.2, random_state=42
    )

    train_loader_b = make_dataloader(X_train_b, y_train_b, batch_size=64, shuffle=True)

    model_b = MLPRegressor(input_dim=X_ohe.shape[1])
    model_b = train_model(model_b,
                          train_loader_b,
                          X_val_b, y_val_b,
                          num_epochs=50, lr=1e-3,
                          device=device,
                          title="2.b With One-Hot")


    print("\n=== 2.c ===")
    model_c = MLPRegressorComplex(input_dim=X_ohe.shape[1])

    train_loader_c = make_dataloader(X_train_b, y_train_b, batch_size=64, shuffle=True)
    model_c = train_model(model_c,
                          train_loader_c,
                          X_val_b, y_val_b,
                          num_epochs=50, lr=1e-3,
                          device=device,
                          title="2.c Larger MLP (One-Hot)")

if __name__ == "__main__":
    main()
