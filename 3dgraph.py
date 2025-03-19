import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demand(ul: float, uf: float):
    return 2 - ul + (0.3 * uf)

def daily_profit(ul: float, uf: float, c: float):
    return (ul - c) * demand(ul, uf)

# import data.xlsx
data_path = "comp34612/data.xlsx"
xls = pd.ExcelFile(data_path)

f1m = pd.read_excel(xls, 'Follower_Mk1')
f2m = pd.read_excel(xls, 'Follower_Mk2')
f3m = pd.read_excel(xls, 'Follower_Mk3')

data = f1m

# Create a 3D graph with axes: Leader price, iteration, Follower price
t = [i for i in range(1, 101)]
leader_price = data.iloc[:, 1]
follower_price = data.iloc[:, 2]
costs = data.iloc[:, 3]

# Plot and export 2D scatter plots of Follower Mk1
plt.scatter(t, leader_price, label="Leader's Price")
plt.scatter(t, follower_price, label="Follower's Price")
plt.xlabel("Iteration")
plt.ylabel("Price")
plt.legend()
plt.title("Follower Mk3")
plt.savefig("graphs/follower_mk1.png")

# Plot leader price vs follower price on new graph
plt.clf()
plt.plot(leader_price, leader_price, label="Leader Price")
plt.scatter(leader_price, follower_price)
plt.xlabel("Leader Price")
plt.ylabel("Follower Price")
plt.title("Leader Price vs Follower Price FM1")
plt.savefig("graphs/fm1_leader_vs_follower.png")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(leader_price, t, follower_price)
ax.set_xlabel("Leader Price")
ax.set_ylabel("Iteration")
ax.set_zlabel("Follower Price")
ax.set_title("3D Graph of Leader Price, Iteration, and Follower Price FM1")

fig.savefig("graphs/fm1_3d_graph.png")

plt.clf()
# Create demand graph
demand_list = []
t = [i for i in range(1, 101)]
for i in range(100):
    demand_list.append(demand(leader_price.iloc[i], follower_price.iloc[i]))

plt.scatter(t, demand_list)
plt.xlabel("Iteration")
plt.ylabel("Follower Price")
plt.title("Daily Demand FM1")
plt.savefig("graphs/fm1_demand.png")

plt.clf()

# Create daily profit graph
profit_list = []
t = [i for i in range(1, 101)]
for i in range(100):
    profit_list.append(daily_profit(leader_price.iloc[i], follower_price.iloc[i], costs.iloc[i]))

plt.scatter(t, profit_list)
plt.xlabel("Iteration")
plt.ylabel("Daily Profit")
plt.title("Daily Profit FM1")
plt.savefig("graphs/fm1_profit.png")

plt.clf()
plt.scatter(leader_price, profit_list)
plt.xlabel("Leader Price")
plt.ylabel("Profit")
plt.title("Leader Price vs Profit FM1")
plt.savefig("graphs/lp_profit_fm1.png")

plt.clf()
plt.scatter(follower_price, demand_list)
plt.xlabel("Follower Price")
plt.ylabel("Demand")
plt.title("Follower Price vs Demand FM1")
plt.savefig("graphs/fp_demand_fm1.png")

plt.clf()
plt.scatter(leader_price, demand_list)
plt.xlabel("Leader Price")
plt.ylabel("Demand")
plt.title("Leader Price vs Demand FM1")
plt.savefig("graphs/lp_demand_fm1.png")

plt.clf()
plt.scatter(follower_price, profit_list)
plt.xlabel("Follower Price")
plt.ylabel("Profit")
plt.title("Follower Price vs Profit FM1")
plt.savefig("graphs/fp_profit_fm1.png")

plt.clf()
delta_fp = [np.abs(follower_price[i-1] - follower_price[i]) for i in range(1, 100)]
plt.scatter(t[1:], delta_fp)
plt.xlabel("Iteration")
plt.ylabel("Delta FP")
plt.title("Iteration vs Delta FP FM1")
plt.savefig("graphs/delta_fp_fm1.png")

plt.clf()
delta_lp = [np.abs(leader_price[i-1] - leader_price[i]) for i in range(1, 100)]
plt.scatter(t[1:], delta_lp)
plt.xlabel("Iteration")
plt.ylabel("Delta LP")
plt.title("Iteration vs Delta LP FM1")
plt.savefig("graphs/delta_lp_fm1.png")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

ax2.scatter(leader_price, follower_price, profit_list)
ax2.set_xlabel("Leader Price")
ax2.set_ylabel("Follower Price")
ax2.set_zlabel("Profit")
ax2.set_title("3D Graph of Leader Price, Follower Price, and Profit FM1")

fig2.savefig("graphs/fm1_lp_fp_prof_3d_graph.png")
plt.show()

print(f"LP-FP Correlation: {np.corrcoef(leader_price, follower_price)[0, 1]}")
print(f"LP-FP Covariance: {np.cov(leader_price, follower_price)[0, 1]}")
print(f"LP-T Correlation: {np.corrcoef(leader_price, t)[0, 1]}")
print(f"LP-T Covariance: {np.cov(leader_price, t)[0, 1]}")
print(f"T-FP Correlation: {np.corrcoef(t, follower_price)[0, 1]}")
print(f"T-FP Covariance: {np.cov(t, follower_price)[0, 1]}")
print(f"LP-Dem Correlation: {np.corrcoef(leader_price, demand_list)[0, 1]}")
print(f"LP-Prof Correlation: {np.corrcoef(leader_price, profit_list)[0, 1]}")
print(f"FP-Dem Correlation: {np.corrcoef(follower_price, demand_list)[0, 1]}")
print(f"FP-Prof Correlation: {np.corrcoef(follower_price, profit_list)[0, 1]}")
print(f"T-Dem Correlation: {np.corrcoef(t, demand_list)[0, 1]}")
print(f"T-Prof Correlation: {np.corrcoef(t, profit_list)[0, 1]}")