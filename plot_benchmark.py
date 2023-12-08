import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_file = sys.argv[1]

data = pd.read_csv(csv_file)

plt.plot(data["iter"], data["y"])
plt.xlabel("Iteration")
plt.ylabel("Obj Fn")
plt.yscale("log")
plt.ylim(10e-2, 10e2)
plt.xlim(0, 200)
plt.show()
