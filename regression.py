import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class regressionPredictor:

    def __init__(self):
        self.models = {}
        
    def train(self, follower_number:int):
        
        if follower_number not in range(1,7):
            raise Exception("Followers must be 1-6 only")
        
        # read data and convert to tensors
        df = pd.read_excel("data.xlsx",sheet_name=f"Follower_Mk{follower_number}")

        x = torch.tensor(df["Leader's Price"].values, dtype=torch.float32).view(-1, 1)
        y = torch.tensor(df["Follower's Price"].values, dtype=torch.float32).view(-1, 1)

        polnomialDegrees = 3 #TODO if this is raised too high, learning rate must be lowered to avoid nan memory overflow due to large gradients
        x_polynomial = torch.cat([x**i for i in range(1, polnomialDegrees+1)], dim=1)

        # define a one layer model with only linear regression of our polynomial weights
        class Model(nn.Module):
            def __init__(self, input_dim):
                super(Model, self).__init__()
                self.linear = nn.Linear(input_dim, 1)
                self.polynomialDegrees = input_dim

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

            if epoch % 5000 == 0:
                print(f"Loss {loss}  Epoch: {epoch}  Model: {follower_number}")
        print(f"\nFinished Training follower {follower_number} \nWith final loss: {loss_model(model(x_polynomial), y)}\n")
        self.models[follower_number] = model

    def trainAllModels(self):
        for i in range(1,4): #TODO handle data from followers 4-6
            self.train(i)

    def getPredictions(self, follower_number:int, data:torch.Tensor):

        if follower_number not in range(1,7):
            raise Exception("Followers must be 1-6 only")
        
        polynomial_data = torch.cat([data**i for i in range(1, self.models[follower_number].polynomialDegrees+1)], dim=1)
        predictions = self.models[follower_number](polynomial_data)
        return predictions
    
    def convertTensorToList(self, tensor:torch.Tensor):
        return tensor.view(-1).tolist()
        


if __name__ == "__main__":
    regression_predictor = regressionPredictor()
    regression_predictor.trainAllModels()

    # example predictions
    follower_number = 1
    df = pd.read_excel("data.xlsx",sheet_name=f"Follower_Mk{follower_number}")
    x = torch.tensor(df["Leader's Price"].values, dtype=torch.float32).view(-1, 1)
    print(regression_predictor.convertTensorToList(regression_predictor.getPredictions(follower_number,x)))