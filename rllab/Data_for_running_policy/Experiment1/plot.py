import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/home/yathartha/PycharmProjects/RL/cassierl/rllab/Data_for_running_policy/Experiment1/progress.csv")
avg_return = data["AverageReturn"]

avg_return = data["AverageReturn"]
StdReturn = data["StdReturn"]
NumTrajs = data["NumTrajs"]
Entropy = data["Entropy"]
AveragePolicyStd = data["AveragePolicyStd"]
MeanKL = data["MeanKL"]
Perplexity = data["Perplexity"]
AverageDiscountedReturn = data["AverageDiscountedReturn"]
dLoss = data["dLoss"]

iter = data["Iteration"]
plt.plot(iter, avg_return)
plt.title("avg return")
plt.show()
#
# iter = data["Iteration"]
# plt.plot(iter, StdReturn)
# plt.title("std return")
# plt.show()
#
# iter = data["Iteration"]
# plt.plot(iter, NumTrajs)
# plt.title("Num Trajs")
# plt.show()
#
# iter = data["Iteration"]
# plt.plot(iter, Entropy)
# plt.title("entropy")
# plt.show()
#
# iter = data["Iteration"]
# plt.plot(iter, AveragePolicyStd)
# plt.title("avg policy std")
# plt.show()
#
# iter = data["Iteration"]
# plt.plot(iter, MeanKL)
# plt.title("mean KL")
# plt.show()
#
# iter = data["Iteration"]
# plt.plot(iter, Perplexity)
# plt.title("Perplexity")
# plt.show()
#
# iter = data["Iteration"]
# plt.plot(iter, AverageDiscountedReturn)
# plt.title("AverageDiscountedReturn")
# plt.show()
#
# iter = data["Iteration"]
# plt.plot(iter, dLoss)
# plt.title("mdLoss")
# plt.show()