import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/home/yathartha/PycharmProjects/RL/cassierl/rllab/Data_for_running_policy/Experiment5/progress.csv")
avg_return = data["AverageReturn"]
iter = data["Iteration"]
plt.plot(iter, avg_return)
plt.show()

