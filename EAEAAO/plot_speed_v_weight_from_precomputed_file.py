import numpy as np
import pickle
import matplotlib.pyplot as plt

input_file = "ALL_WEIGHTS_AND_SPEED_MEAN_IN_PX_per_S.pickle"

with open(input_file, 'rb') as pickle_file:
    weight_v_speed = pickle.load(pickle_file)

unique_classes = np.unique(weight_v_speed[:, 0])

sorted_weight_v_speed = [[] for i in range(len(unique_classes))]

# see I:\EAEAAO\CALIBRATION\2019-07-23_bramble_right\px_to_cm_conversion.xlsx

px_to_mm = 6.8628

for elem in weight_v_speed:
    sorted_weight_v_speed[int(elem[0])].append(float(elem[1]) / px_to_mm)

for c, speed_list in enumerate(sorted_weight_v_speed):
    print(c, "contains", len(speed_list), "individuals")

plt.boxplot(sorted_weight_v_speed)

plt.title("mean speed observed for each size class")
plt.xlabel("size class")
plt.xticks([1, 2, 3, 4, 5], ["0.0013", "0.0030", "0.0068", "0.0154", "0.0351"])
plt.ylabel("speed in mm/s")
plt.ylim((0, 30))

text = f"n = " + str(len(weight_v_speed))

plt.gca().text(0.75, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

plt.savefig("Size-speed distribution (mean) plot (A).svg")

plt.show()

# plot the speed distribution within the 0.003 g weight class

color = (0.2,  # redness
         0.1,  # greenness
         0.9,  # blueness
         0.6  # transparency
         )

plt.hist(sorted_weight_v_speed[1], bins=range(9, 17), density=True,
         color=color)

plt.title("speed distribution within the 0.003 g worker class")
plt.xlabel("speed in mm/s")
plt.ylabel("frequency")

text = f"n = " + str(len(sorted_weight_v_speed[1]))

plt.gca().text(0.75, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

plt.savefig("Size-speed distribution (0.003 g, mean) plot (C).svg")

plt.show()
