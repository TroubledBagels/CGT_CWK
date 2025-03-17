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

used_data = fm1
used_data = remove_outliers(used_data)

# Plot follower price vs leader price
plt.scatter(used_data.iloc[:, 1], used_data.iloc[:, 2])
plt.xlabel("Leader Price")
plt.ylabel("Follower Price")
plt.title("Follower Mk1 Dummy")

# Find follower price as a function of leader price
from sklearn.linear_model import LinearRegression
X = used_data.iloc[:, 1].values.reshape(-1, 1)
y = used_data.iloc[:, 2].values
reg = LinearRegression().fit(X, y)
print(f"R^2: {reg.score(X, y)}")
print(f"Intercept: {reg.intercept_}")
print(f"Coefficient: {reg.coef_}")

# Use quadratic regression
from sklearn.preprocessing import PolynomialFeatures
square = PolynomialFeatures(degree=2)
X_square = square.fit_transform(X)
reg_square = LinearRegression().fit(X_square, y)
print(f"R^2: {reg_square.score(X_square, y)}")
print(f"Intercept: {reg_square.intercept_}")
print(f"Coefficient: {reg_square.coef_}")

# Use cubic regression
cubic = PolynomialFeatures(degree=3)
X_cubic = cubic.fit_transform(X)
reg_cubic = LinearRegression().fit(X_cubic, y)
print(f"R^2: {reg_cubic.score(X_cubic, y)}")
print(f"Intercept: {reg_cubic.intercept_}")
print(f"Coefficient: {reg_cubic.coef_}")

# Use quartic regression
quartic = PolynomialFeatures(degree=4)
X_quartic = quartic.fit_transform(X)
reg_quartic = LinearRegression().fit(X_quartic, y)
print(f"R^2: {reg_quartic.score(X_quartic, y)}")
print(f"Intercept: {reg_quartic.intercept_}")
print(f"Coefficient: {reg_quartic.coef_}")

# Use quintic regression
quintic = PolynomialFeatures(degree=5)
X_quintic = quintic.fit_transform(X)
reg_quintic = LinearRegression().fit(X_quintic, y)
print(f"R^2: {reg_quintic.score(X_quintic, y)}")
print(f"Intercept: {reg_quintic.intercept_}")
print(f"Coefficient: {reg_quintic.coef_}")

# Plot the regression lines
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Transform these values for polynomial predictions
X_square_range = square.transform(X_range)
X_cubic_range = cubic.transform(X_range)
X_quartic_range = quartic.transform(X_range)
X_quintic_range = quintic.transform(X_range)

# Plot the original scatter plot
plt.scatter(X, y, color='black', label="Data")

# Plot regression lines
plt.plot(X_range, reg.predict(X_range), color='red', label="Linear")
plt.plot(X_range, reg_square.predict(X_square_range), color='green', label="Quadratic")
plt.plot(X_range, reg_cubic.predict(X_cubic_range), color='blue', label="Cubic")
plt.plot(X_range, reg_quartic.predict(X_quartic_range), color='purple', label="Quartic")
plt.plot(X_range, reg_quintic.predict(X_quintic_range), color='orange', label="Quintic")

plt.xlabel("Leader Price")
plt.ylabel("Follower Price")
plt.title("Polynomial Regression for Follower Mk1")
plt.legend()
plt.show()