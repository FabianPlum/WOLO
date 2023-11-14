import numpy as np
import pickle
import matplotlib.pyplot as plt

input_file = "ALL_WEIGHTS_AND_SPEED_IN_PX_per_S.pickle"
# input_file = "2019-07-22_bramble_left_WEIGHTS_AND_SPEED_IN_PX_per_S.pickle"

with open(input_file, 'rb') as pickle_file:
    weight_v_speed = pickle.load(pickle_file)

unique_classes = np.unique(weight_v_speed[:, 0])

sorted_weight_v_speed = [[] for i in range(len(unique_classes))]

px_to_cm = 1000 / 20  # recorded area is roughly 15 cm long and spans 1000 px in the image

for elem in weight_v_speed:
    sorted_weight_v_speed[int(elem[0])].append(float(elem[1]) / px_to_cm)

for c, speed_list in enumerate(sorted_weight_v_speed):
    print(c, "contains", len(speed_list), "individuals")

plt.boxplot(sorted_weight_v_speed)

plt.title("maximum speed observed for each size class")
plt.xlabel("size class")
plt.xticks([1, 2, 3, 4, 5], ["0.0013", "0.0030", "0.0068", "0.0154", "0.0351"])
plt.ylabel("maximum speed in cm/s")
plt.ylim((0, 30))

text = f"n = " + str(len(weight_v_speed))

plt.gca().text(0.75, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

plt.savefig("Size-speed distribution plot (A).svg")

plt.show()

weight_v_speed_mean = [np.mean(w) for w in sorted_weight_v_speed]
weight_v_speed_std = [np.std(w) for w in sorted_weight_v_speed]

plt.bar([1, 2, 3, 4, 5], weight_v_speed_mean)

plt.title("maximum speed observed for each size class")
plt.xlabel("size class")
plt.xticks([1, 2, 3, 4, 5], ["0.0013", "0.0030", "0.0068", "0.0154", "0.0351"])
plt.ylabel("maximum speed in cm/s")
plt.ylim((0, 30))
plt.errorbar([1, 2, 3, 4, 5], weight_v_speed_mean, weight_v_speed_std, fmt='.', color='Black', elinewidth=2,
             capthick=10, errorevery=1, alpha=1, ms=4, capsize=2)

text = f"n = " + str(len(weight_v_speed))

plt.gca().text(0.75, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

plt.savefig("Size-speed distribution plot (B).svg")

plt.show()
