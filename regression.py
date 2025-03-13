import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# read data and convert to tensors
df = pd.read_excel("data.xlsx",sheet_name="Follower_Mk1")
print(df)

x = torch.tensor(df["Leader's Price"].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(df["Follower's Price"].values, dtype=torch.float32).view(-1, 1)

polnomialDegrees = 3
x_polynomial = torch.cat([x**i for i in range(1, polnomialDegrees+1)], dim=1)
print(x_polynomial[:4])


# define a one layer model with only linear regression of our polynomial weights
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
model = Model(polnomialDegrees)


# define loss/optimiser and train
loss_model = nn.MSELoss()
optimiser = optim.SGD(model.parameters(), lr=0.01)
epochs = 25_000
for epoch in range(epochs):
    
    model.train()
    
    #zero gradients to stop carry over across loops
    optimiser.zero_grad()

    predictions = model(x_polynomial)
    loss = loss_model(predictions, y)

    loss.backward()
    optimiser.step()

    if epoch % 1000 == 0:
        print(f"Loss {loss}  Epoch: {epoch}")



#view sample predictions
all_predictions = model(x_polynomial)
for i,prediction in enumerate(predictions[:10]):
    print(f"prediction: {prediction[0].item()}  True: {y[i].item()}")

print(f"\nFinal loss: {loss_model(all_predictions, y)}")