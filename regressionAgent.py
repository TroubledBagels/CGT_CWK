import gc

class Leader:
    _subclass_registry = {}

    def __init__(self, name, engine):
        self.name = name
        self.engine = engine

    @classmethod
    def cleanup_old_subclasses(cls):
        """
        A function to remove old subclasses before defining new ones.
        """
        existing_subclasses = list(cls.__subclasses__())

        for subclass in existing_subclasses:
            subclass_name = subclass.__name__
            if subclass_name in cls._subclass_registry:
                del cls._subclass_registry[subclass_name]
                del subclass
        gc.collect()

    @classmethod
    def update_subclass_registry(cls):
        """
        A function to update registry after cleaning up old subclasses.
        """
        cls.cleanup_old_subclasses()
        cls._subclass_registry = {subclass.__name__: subclass for subclass in cls.__subclasses__()}

    def new_price(self, date):
        """
        A function for setting the new price of each day.
        :param date: date of the day to be updated
        :return: (float) price for the day
        """
        pass

    def get_price_from_date(self, date):
        """
        A function for getting the price set on a date.
        :param date: (int) date to get the price from
        :return: a tuple (leader_price, follower_price)
        """
        return self.engine.exposed_get_price(date)


    def start_simulation(self):
        """
        A function runs at the beginning of the simulation.
        """
        pass

    def end_simulation(self):
        """
        A function runs at the beginning of the simulation.
        """
        pass


# Copy paste the below chunk into the notebook in the revelant cell

from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import numpy as np
class RegressionLeader(Leader):

    leader_price_history = []
    follower_price_history = []

    def __init__(self, name, engine):
        super().__init__(name, engine)
    
    def new_price(self, date:int):

        dates = [i for i in range(1,(len(self.leader_price_history)+1))]
        self.model = LinearRegression().fit(np.column_stack((self.leader_price_history, dates)), self.follower_price_history) #takes price AND date as features

        result = minimize(self.get_profit, x0=self.leader_price_history[-1], args=date, bounds=[(1,100)]) #TODO change bounds depending on follower

        #TODO append result of current day to history, to use in the following days

        return result.x[0]
    
    def start_simulation(self):

        self.leader_price_history = []
        self.follower_price_history = []

        #load past 100 days of example data
        for i in range(1,101):
            leaderPrice, followerPrice = self.get_price_from_date(i)
            self.leader_price_history.append(leaderPrice)
            self.follower_price_history.append(followerPrice)

    def get_profit(self, leader_x, date):

        follower_x = self.model.predict(np.array([[leader_x[0],date]]))[0] 
        sales = 2 - leader_x + (0.3*follower_x)
        profit = (leader_x - 1) * sales
        return -profit #as we are minimizing
