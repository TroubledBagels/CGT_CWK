import numpy as np
from scipy.optimize import minimize
         
class StackelbergSolver:
    def __init__(self, follower_model): 

        self.follower_model = follower_model #assumes this model has a .predict() that takes a numpy array
        self.x_bounds = (1,2) #this changes depending on model

    def get_profit(self, leader_x):

        follower_x = self.follower_model.predict(np.array([leader_x]))[0] 
        sales = 2 - leader_x + (0.3*follower_x)
        profit = (leader_x - 1) * sales
        return profit

    def solve(self, x0 = None):
   
        if x0 is None:
            x0 = np.mean(self.x_bounds) 
        
        result = minimize(self.get_profit, x0, bounds=[self.x_bounds])
        best_x = result.x[0]

        return best_x
    

# example use
if __name__ == "__main__":
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    df = pd.read_excel("data.xlsx",sheet_name=f"Follower_Mk{3}")
    
    follower_model = LinearRegression().fit(df["Leader's Price"].values.reshape(-1, 1), df["Follower's Price"].values)

    solver = StackelbergSolver(follower_model)
    print(f"Best X is: {solver.solve()}")