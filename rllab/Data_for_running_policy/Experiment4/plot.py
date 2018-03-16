import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("progress.csv")
avg_return = data["AverageReturn"]
iter = data["Iteration"]
plt.plot(iter, avg_return)
plt.show()

