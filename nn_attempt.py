import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np


class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.features = None
        self.labels = None

        self.convert_to_tensor(data)

    def convert_to_tensor(self, data: pd.DataFrame):
        # Data in form of [Date, Leader Price, Follower Price (Label)]
        self.labels = data.iloc[:, 2].values
        self.features = data.iloc[:, 0:2].values
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()
        self.input_size = 2

        self.fc1 = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class NNLeader():
    def __init__(self, name, engine):
        xls = pd.ExcelFile("comp34612/data.xlsx")
        self.data = pd.read_excel(xls, 'Follower_Mk1')
        self.dataset = PriceDataset(self.data)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.follower_prices = self.data.iloc[:, 2]
        self.features = self.data.iloc[:, 0:2]
        self.model = PricePredictor()
        self.train()

    def train(self):
        optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(50):
            for x, y in self.dataloader:
                optimiser.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimiser.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def calc_demand(self, ul: float, uf: float):
        return 2 - ul + 0.3 * uf

    def calc_profit(self, ul: float, uf: float, cl: float = 1):
        return (ul - cl) * self.calc_demand(ul, uf)

    def new_price(self, date: int):
        max_l_price = np.max(self.data.iloc[:, 1])
        min_l_price = np.min(self.data.iloc[:, 1])
        max_limit = max_l_price + 0.5
        min_limit = min_l_price - 0.5
        print(f"Min limit: {min_limit}, Max limit: {max_limit}")

        cur_best = (0, 0)

        self.model.eval()
        with torch.no_grad():
            test_val = min_limit
            while test_val <= max_limit:
                input_val = torch.tensor((date, test_val), dtype=torch.float32)
                follower_pred = self.model(input_val)

                new_profit = self.calc_profit(test_val, follower_pred.item())

                if new_profit > cur_best[1]:
                    cur_best = (test_val, new_profit)

                test_val += 0.01

        return cur_best[0]

leader = NNLeader(None, None)
print(leader.new_price(101))