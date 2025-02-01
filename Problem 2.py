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

file_path = 'https://raw.githubusercontent.com/HamedTabkhi/Intro-to-ML/refs/heads/main/Dataset/Housing.csv'
df = pd.read_csv(file_path)

df['price'] = df['price'].astype(np.float32) / 1e6

def prepare_data_no_ohe(df):

    df_copy = df.copy()
    y = df_copy['price'].values
    df_copy.drop('price', axis=1, inplace=True)

    yes_no_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
    for c in yes_no_cols:
        df_copy[c] = df_copy[c].map({'yes':1,'no':0}).astype(np.float32)

    f_map = {'furnished':2, 'semi-furnished':1, 'unfurnished':0}
    df_copy['furnishingstatus'] = df_copy['furnishingstatus'].map(f_map).astype(np.float32)

    X = df_copy.values.astype(np.float32)
    return X, y

def prepare_data_ohe(df):
    df_copy = df.copy()
    y = df_copy['price'].values
    df_copy.drop('price', axis=1, inplace=True)

    yes_no_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
    for c in yes_no_cols:
        df_copy[c] = df_copy[c].map({'yes':1,'no':0}).astype(np.float32)

    fstatus = df_copy['furnishingstatus']
    df_copy.drop('furnishingstatus', axis=1, inplace=True)
    fstatus_ohe = pd.get_dummies(fstatus, prefix='fstatus')

    df_ohe = pd.concat([df_copy, fstatus_ohe], axis=1)
    X = df_ohe.values.astype(np.float32)
    return X, y

def create_dataloader(X, y, batch_size=64, shuffle=True):
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).view(-1,1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
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
        super().__init__()
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

def train_model(model, train_loader, X_test, y_test,
                num_epochs=50, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).view(-1,1).to(device)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            num_samples += batch_x.size(0)
        epoch_loss = running_loss / num_samples
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
            val_loss = criterion(pred_test, y_test_t).item()
        test_losses.append(val_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train MSE: {epoch_loss:.4f}, Test MSE: {val_loss:.4f}")

    # 画图
    plt.figure()
    plt.plot(train_losses, label='Train MSE')
    plt.plot(test_losses, label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title("Training & Test MSE")
    plt.show()

    # 计算RMSE, R^2
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()
    mse_val = mean_squared_error(y_test, preds)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_test, preds)
    print("RMSE:", rmse_val)
    print("R^2:", r2_val)

    return model

#--------------------------------------------------------------------------
# 4. main
#--------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #================= 2.a 不使用 One-Hot =================
    print("\n=== 2.a ")
    X_no_ohe, y_no_ohe = prepare_data_no_ohe(df)

    # 特征标准化(可选，但建议)
    scaler_no_ohe = StandardScaler()
    X_no_ohe = scaler_no_ohe.fit_transform(X_no_ohe)

    X_train_no_ohe, X_test_no_ohe, y_train_no_ohe, y_test_no_ohe = train_test_split(
        X_no_ohe, y_no_ohe, test_size=0.2, random_state=42
    )
    train_loader_no_ohe = create_dataloader(X_train_no_ohe, y_train_no_ohe, batch_size=64)
    model_no_ohe = MLPRegressor(input_dim=X_no_ohe.shape[1])
    _ = train_model(model_no_ohe, train_loader_no_ohe,
                    X_test_no_ohe, y_test_no_ohe,
                    num_epochs=50, device=device)

    print("\n=== 2.b ===")
    X_ohe, y_ohe = prepare_data_ohe(df)

    scaler_ohe = StandardScaler()
    X_ohe = scaler_ohe.fit_transform(X_ohe)

    X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(
        X_ohe, y_ohe, test_size=0.2, random_state=42
    )
    train_loader_ohe = create_dataloader(X_train_ohe, y_train_ohe, batch_size=64)
    model_ohe = MLPRegressor(input_dim=X_ohe.shape[1])
    _ = train_model(model_ohe, train_loader_ohe,
                    X_test_ohe, y_test_ohe,
                    num_epochs=50, device=device)

    print("\n=== 2.c ===")
    model_complex = MLPRegressorComplex(input_dim=X_ohe.shape[1])
    train_loader_c = create_dataloader(X_train_ohe, y_train_ohe, batch_size=64)
    _ = train_model(model_complex, train_loader_c,
                    X_test_ohe, y_test_ohe,
                    num_epochs=50, device=device)

if __name__ == "__main__":
    main()
