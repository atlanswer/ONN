# import sys
import pandas as pd
import matplotlib.pyplot as plt

# csv_file = sys.argv[1]

run1 = pd.read_csv(r".\save\12082119.csv")
run2 = pd.read_csv(r".\save\12100111.csv")


plt.style.use(["default", "seaborn-v0_8-paper", "./publication.mplstyle"])
plt.figure(figsize=(3.5, 2))
plt.plot(run1["iter"], run1["loss"], "b-")
plt.plot(run2["iter"], run2["loss"], "b--")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.ylim(10e-5, 10e5)
plt.xlim(0, 200)
# plt.show()
plt.savefig("fig.svg", bbox_inches="tight")
