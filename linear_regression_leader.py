import numpy as np
import scipy.optimize as opt
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

class RegressionLeader(Leader):
    def __init__(self, name, engine):
        super().__init__(name, engine)

    def leader_profit(self, ul: float, cl: float, follower_intercept: float, follower_gradient: float):
        demand = 2 - ul + 0.3 * (follower_intercept + follower_gradient * ul)
        profit = (ul - cl) * demand
        return -profit

    def calc_demand(self, ul: float, uf: float):
        return 2 - ul + 0.3 * uf

    def calc_profit(self, ul: float, uf: float, cl: float = 1):
        return (ul - cl) * self.calc_demand(ul, uf)

    def stackel_solve(self, reaction_coef: tuple[float, float], bounds: tuple[float, float]):
        follower_gradient = reaction_coef[0]
        follower_intercept = reaction_coef[1]

        leader_cost = 1

        result = opt.minimize_scalar(self.leader_profit, bounds=bounds, args=(leader_cost, follower_intercept, follower_gradient))

        ul_star = result.x

        uf_star = follower_intercept + follower_gradient * ul_star

        profit_l_star = self.calc_profit(ul_star, uf_star, leader_cost)

        return ul_star, uf_star

    def new_price(self, date: int):
        pass