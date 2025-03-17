import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Plot and export 2D scatter plots of Follower Mk1
plt.scatter(t, leader_price, label="Leader's Price")
plt.scatter(t, follower_price, label="Follower's Price")
plt.xlabel("Iteration")
plt.ylabel("Price")
plt.legend()
plt.title("Follower Mk1")
plt.savefig("comp34612/follower_mk1.png")

# Plot leader price vs follower price on new graph
plt.clf()
plt.scatter(leader_price, follower_price)
plt.xlabel("Leader Price")
plt.ylabel("Follower Price")
plt.title("Leader Price vs Follower Price")
plt.savefig("comp34612/leader_vs_follower.png")

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(leader_price, t, follower_price)
ax.set_xlabel("Leader Price")
ax.set_ylabel("Iteration")
ax.set_zlabel("Follower Price")

fig.savefig("comp34612/3d_graph.png")

plt.show()

print(f"LP-FP Correlation: {np.corrcoef(leader_price, follower_price)[0, 1]}")
print(f"LP-FP Covariance: {np.cov(leader_price, follower_price)[0, 1]}")
print(f"LP-T Correlation: {np.corrcoef(leader_price, t)[0, 1]}")
print(f"LP-T Covariance: {np.cov(leader_price, t)[0, 1]}")
print(f"T-FP Correlation: {np.corrcoef(t, follower_price)[0, 1]}")
print(f"T-FP Covariance: {np.cov(t, follower_price)[0, 1]}")
