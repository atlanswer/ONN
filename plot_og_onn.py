import sys
import csv
import matplotlib.pyplot as plt
# import numpy as np

csv_file = sys.argv[1]

# Load the data from the CSV file
with open(csv_file, "r") as f:
    reader = csv.reader(f, skipinitialspace=True)
    x = []
    y = []
    for row in reader:
        if row:  # check that row is not empty
            x.append(float(row[0]))
            y.append(float(row[-1]))

# Plot the data
plt.plot(x, y)
plt.xlabel("Iteration")
plt.ylabel("Obj Fn")
plt.yscale('log')
plt.ylim(10e-2, 10e2)
plt.xlim(0, 200)
plt.show()
