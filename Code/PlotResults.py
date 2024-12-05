from matplotlib import pyplot as plt

pop_size_list = [30, 60, 90, 150, 240, 360, 450]

Sim1 = [11.65333333, 17.07, 23.37, 36.15666667, 58.59666667, 86.84666667, 104.34333333]
Sim2 = [37.77333333, 37.74666667, 40.72333333, 41.99666667, 46.83333333, 67.97666667, 83.94666667]
Sim3 = [ 23.19333333, 22.86333333, 23.36333333, 38.13333333, 57.99, 86.83, 106.51666667]

plt.plot(pop_size_list, Sim1, label="Closest Exit")
plt.plot(pop_size_list, Sim2, label="Assigned Exit")
plt.plot(pop_size_list, Sim3, label="Closest Exit + Some drunk agents")
plt.ylabel("Total Evacuation Time (s)")
plt.xlabel("Number of agents")
plt.title("Total evacuation time for different number of agents.")
plt.text(20,85,"v0 = 4, Exits/s = 2.")
plt.legend()
plt.show()