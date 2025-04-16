import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data.xlsx
data_path = "comp34612/data.xlsx"
xls = pd.ExcelFile(data_path)
noise = pd.read_excel(xls, 'Test_Noises')

print(noise["Mk1"][:30])

# Plot the noise
days = [i for i in range(101, 131)]
print(len(days))
plt.scatter(days, noise["Mk1"][:30])
plt.xlabel("Time")
plt.ylabel("Noise")
plt.title("Noise")
plt.show()
