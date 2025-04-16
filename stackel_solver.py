import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt

def leader_profit(ul: float, cl: float, follower_intercept: float, follower_gradient: float):
    demand = 2 - ul + 0.3 * (follower_intercept + follower_gradient * ul)
    profit = (ul - cl) * demand
    return -profit

def calc_demand(ul: float, uf: float):
    return 2 - ul + 0.3 * uf

def calc_profit(ul: float, uf: float, cl: float = 1):
    return (ul - cl) * calc_demand(ul, uf)

def stackel_solve(reaction_coef: list[float], bounds: tuple[float, float]):
    # Reaction coefficients in form of [0, x^(len-2), ..., x^2, x^1, x^0]
    # Demand in form of [c, uf_mult, ul_mult]
    # Then put in form: demand = c + ul_mult * ul + uf_mult * uf
    follower_intercept = reaction_coef[1]
    follower_gradient = reaction_coef[0]

    demand_intercept = 2
    demand_sensitivity_follower = 0.3
    demand_sensitivity_leader = -1

    leader_cost = 1

    # demand = demand_intercept + demand_sensitivity_leader * ul + demand_sensitivity_follower * uf
    # = 2 - ul + 0.3 * uf
    # = 2 - ul + 0.3 * (follower_intercept + follower_gradient * ul)
    # profit = (ul - leader_cost) * demand

    result = opt.minimize_scalar(leader_profit, bounds=bounds, args=(leader_cost, follower_intercept, follower_gradient))

    ul_star = result.x

    uf_star = follower_intercept + follower_gradient * ul_star

    profit_l_star = calc_profit(ul_star, uf_star, leader_cost)

    print(f"Leader's optimal price: {ul_star}")
    print(f"Follower's optimal price: {uf_star}")
    print(f"Leader's profit: {profit_l_star}")
    return ul_star, uf_star

def remove_outliers(data):
    # Find the mean and standard deviation of the data
    mean = data.mean()
    std = data.std()

    # Find the lower and upper bounds
    lower_bound = mean - 1.5 * std
    upper_bound = mean + 1.5 * std

    new_data = pd.DataFrame(columns=data.columns)

    # Remove the outliers
    print(data)
    for i in range(len(data)):
        if upper_bound[2] >= data.iloc[i, 2] >= lower_bound[2]:
            new_data = new_data._append(data.iloc[i])
    return new_data

def main():
    data_path = "comp34612/data.xlsx"
    xls = pd.ExcelFile(data_path)
    fm1d = pd.read_excel(xls, 'Follower_Mk1_Dummy')
    fm1 = pd.read_excel(xls, 'Follower_Mk1')
    fm2d = pd.read_excel(xls, 'Follower_Mk2_Dummy')
    fm2 = pd.read_excel(xls, 'Follower_Mk2')
    fm3d = pd.read_excel(xls, 'Follower_Mk3_Dummy')
    fm3 = pd.read_excel(xls, 'Follower_Mk3')
    leader_vars = pd.read_excel(xls, 'Leader Variables')
    test_noises = pd.read_excel(xls, 'Test_Noises')

    used_data = fm2
    used_data = remove_outliers(used_data)

    leader_data = used_data.iloc[:, 1]
    follower_data = used_data.iloc[:, 2]

    R_estimate = np.polyfit(leader_data, follower_data, 1)

    print(f"Reaction coefficients: {R_estimate}")

    plt.scatter(leader_data, follower_data)
    plt.xlabel("Leader Price")
    plt.ylabel("Follower Price")
    plt.title("Follower Mk3")

    plt.plot(leader_data, R_estimate[0] * leader_data + R_estimate[1], color="red")
    plt.show()

    stackel_solve(R_estimate, (0, 2))

if __name__ == "__main__":
    main()
