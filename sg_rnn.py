import torch
import torch.nn as nn

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import random

class SGRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SGRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

# import data.xlsx
data_path = "comp34612/data.xlsx"
xls = pd.ExcelFile(data_path)

f1m = pd.read_excel(xls, 'Follower_Mk1')
f2m = pd.read_excel(xls, 'Follower_Mk2')
f3m = pd.read_excel(xls, 'Follower_Mk3')

data = f1m

def run_rnn(data, hidden_size, epochs):
    t = [i for i in range(1, 101)]
    leader_price = data.iloc[:, 1]
    follower_price = data.iloc[:, 2]

    model = SGRNN(1, hidden_size, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



    for epoch in range(epochs):
        # use 3-fold cross validation
        k1, k2, k3 = np.split(data.sample(frac=1, random_state=random.randint(0,1000)), [int(.33*len(data)), int(.66*len(data))])

        for i in range(3):
            if i == 0:
                train = pd.concat([k2, k3])
                test = pd.DataFrame(k1)
            elif i == 1:
                train = pd.concat([k1, k3])
                test = pd.DataFrame(k2)
            else:
                train = pd.concat([k1, k2])
                test = pd.DataFrame(k3)

            x = torch.tensor(train.iloc[:, 1].values).view(-1, 1, 1)
            y = torch.tensor(train.iloc[:, 2].values).view(-1, 1, 1)

            optimizer.zero_grad()
            output = model(x.float())
            loss = criterion(output, y.float())
            loss.backward()
            optimizer.step()

            # test on test set
            x = torch.tensor(test.iloc[:, 1].values).view(-1, 1, 1)
            y = torch.tensor(test.iloc[:, 2].values).view(-1, 1, 1)
            output = model(x.float())
            loss = criterion(output, y.float())

            error_rate = (output - y) / y
            print(f"Error rate (mean): {error_rate.mean()}")

        # print(f"Epoch {epoch}: Loss {loss.item()}")

    return model

model = run_rnn(data, 10, 40)

# Run again

test_values = [1.8, 1.85, 1.73, 1.775, 1.81, 1.79, 1.77, 1.83, 1.82, 1.81, 1.76, 1.77, 1.78]
y_list = []

for i in range(len(test_values)):
    x = torch.tensor(test_values[i]).view(-1, 1, 1)
    y = model(x.float())
    print(f"Predicted value for {test_values[i]}: {y}")
    y_list.append(y.detach().numpy())

# Plot the original leader price vs follower price
plt.scatter(data.iloc[:, 1], data.iloc[:, 2], label="Original Data")
plt.scatter(test_values, y_list, label="Predicted Data")
plt.xlabel("Leader Price")
plt.ylabel("Follower Price")
plt.title("Leader Price vs Follower Price FM1")
plt.legend()
plt.show()
