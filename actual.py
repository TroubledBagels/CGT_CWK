import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import data.xlsx
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

# For Follower 1
# # Plot and show scatter plot of Follower Mk1
# t = [i for i in range(1, 101)]
# plt.scatter(t, fm1.iloc[:, 1], label="Leader's Price")
# plt.scatter(t, fm1.iloc[:, 2], label="Follower's Price")
# plt.xlabel("t")
# plt.ylabel("Price")
# plt.legend()
# plt.title("Follower Mk1 Dummy")
# plt.draw()
# plt.pause(0.001)
#
# # Plot difference between leader and follower prices
# plt.plot(t, fm1.iloc[:, 1] - fm1.iloc[:, 2])
# plt.xlabel("t")
# plt.ylabel("Price Difference")
# plt.title("Price Difference between Leader and Follower Mk1")
# plt.draw()
# plt.pause(0.001)

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

used_data = fm2
used_data = remove_outliers(used_data)

# Plot follower price vs leader price
plt.scatter(used_data.iloc[:, 1], used_data.iloc[:, 2])
plt.xlabel("Leader Price")
plt.ylabel("Follower Price")
plt.title("Follower Mk1 Dummy")

def calculate_R_estimate(data):
    # Calculate the estimate of the reaction function for the Stackelberg game
    leader_data = data.iloc[:, 1]
    follower_data = data.iloc[:, 2]

    # Calculate the estimate of the reaction function
    R_estimate = np.polyfit(leader_data, follower_data, 1)
    return R_estimate

R_estimate = calculate_R_estimate(used_data)
# Plot the reaction function
plt.plot(used_data.iloc[:, 1], np.polyval(R_estimate, used_data.iloc[:, 1]), color='red')
# plt.plot(used_data.iloc[:, 0], used_data.iloc[:, 2])
plt.show()

# Print r^2 value
leader_data = used_data.iloc[:, 1]
follower_data = used_data.iloc[:, 2]
r_squared = np.corrcoef(leader_data, follower_data)[0, 1] ** 2
print(f"R^2 value: {r_squared}")
