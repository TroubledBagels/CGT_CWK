


# Copy paste the below chunk into the notebook in the revelant cell

from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class RegressionLeader(Leader):

    leader_price_history = []
    follower_price_history = []

    def __init__(self, name, engine):
        super().__init__(name, engine)
    
    def new_price(self, date:int):
        
        # if applicable, add previous day to our dataset
        if date > 101:
            leaderPrice, followerPrice = self.get_price_from_date(date-1)
            self.leader_price_history.append(leaderPrice)
            self.follower_price_history.append(followerPrice)

        # use multi regression on (leader_price, date) to model follower function
        # use date too since some models change reaction over time
        dates = [i for i in range(1,(len(self.leader_price_history)+1))]
        x = np.column_stack((self.leader_price_history, dates))
        y = self.follower_price_history

        # try different degrees and pick best
        max_degree = 4
        best_loss = -100000
        best_model = None
        for degree in range(1, max_degree+1): 
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            losses = cross_val_score(model, x, y)
            avg_loss = np.mean(losses)
            if avg_loss > best_loss:
                best_loss = avg_loss
                best_model = model

        # fit best degree to the dataset
        self.model = best_model.fit(x,y)
        
        # find the leader price that maximses profit, based off the estimate follower reaction
        result = minimize(self.get_profit, x0=self.leader_price_history[-1], args=date, bounds=[(1,2)])     #TODO change bounds depending on follower
        return result.x[0]
    
    def start_simulation(self):

        # reset to avoid using data from last simulation
        self.leader_price_history = []
        self.follower_price_history = []

        #load the given 100 days of example data
        for i in range(1,101):
            leaderPrice, followerPrice = self.get_price_from_date(i)
            self.leader_price_history.append(leaderPrice)
            self.follower_price_history.append(followerPrice)

    def get_profit(self, leader_x, date):

        # formulas from instruction sheet
        follower_x = self.model.predict(np.array([[leader_x[0],date]]))[0] 
        sales = 2 - leader_x + (0.3*follower_x)
        profit = (leader_x - 1) * sales
        return -profit #return negative as we minimize not maximise profit function
