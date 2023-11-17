import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# as we are going to plot all length estimates of all animals in all videos, we can use a singular list of all relevant
# videos, assuming they all have roughly the same magnification and frame of view (which is reasonable here)

all_bramble = ["2019-07-22_bramble_left",
               "2019-07-23_bramble_right2",
               "2019-07-23_bramble_right",
               "2019-07-24_bramble_left",
               "2019-07-24_bramble_right",
               "2019-07-30_bramble_left",
               "2019-07-31_bramble_left",
               "2019-07-31_bramble_right",
               "2019-08-01_bramble_left",
               "2019-08-03_bramble-left",
               "2019-08-03_bramble-right",
               "2019-08-05_bramble_left",
               "2019-08-06_bramble_left",
               "2019-08-07_bramble_left",
               "2019-08-07_bramble_right",
               "2019-08-09_bramble_left",
               "2019-08-13_bramble_right",
               "2019-08-15_bramble_left",
               "2019-08-15_bramble_right",
               "2019-08-16_bramble_right",
               "2019-08-22_bramble_right"]

all_rose = ["2019-07-23_rose_left_2",
            "2019-07-23_rose_left",
            "2019-07-25_rose_left",
            "2019-07-25_rose_right",
            "2019-07-30_rose_right",
            "2019-08-01_rose_right",
            "2019-08-05_rose_right",
            "2019-08-06_rose_right",
            "2019-08-08_rose_left",
            "2019-08-08_rose_right",
            "2019-08-09_rose_right",
            "2019-08-12_rose_left",
            "2019-08-12_rose_right",
            "2019-08-13_rose_left",
            "2019-08-16_rose_left",
            "2019-08-20_rose_left",
            "2019-08-20_rose_right",
            "2019-08-21_rose_left",
            "2019-08-21_rose_right",
            "2019-08-22_rose_left"]

pose_absolute_path = "I:/EAEAAO/POSES"

# NOTE: the pose conversion is larger, as we upscale the extracted patches to the training resolution
# of the DLC networks (from 128 px to 300 px)
px_to_mm = 16.0848

all_lengths_BRAMBLE = []
all_lengths_ROSE = []

for video in all_bramble:
    path = os.path.join(pose_absolute_path, video)
    print("Processing:", path)
    for r, d, f in os.walk(path):
        for file in f:
            df = pd.read_csv(os.path.join(path, file), delimiter=',', header=[0, 1, 2])

            x_diff = df["OmniTrax"]["b_t"]["x"].to_numpy() - df["OmniTrax"]["b_a_5"]["x"].to_numpy()
            y_diff = df["OmniTrax"]["b_t"]["y"].to_numpy() - df["OmniTrax"]["b_a_5"]["y"].to_numpy()
            lengths = np.sqrt(np.square(x_diff) + np.square(y_diff))
            median_length = np.round(np.median(lengths) / px_to_mm, 2)
            all_lengths_BRAMBLE.append(median_length)

plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

green = (0.2,  # redness
         0.4,  # greenness
         0.2,  # blueness
         0.6  # transparency
         )

pink = (0.9,  # redness
        0,  # greenness
        0.2,  # blueness
        0.6  # transparency
        )

# Plot Histogram on x
plt.hist(all_lengths_BRAMBLE, bins=[2.5, 3, 3.5, 4, 4.5, 5], density=True, color=green)
# ax.set_xticks(np.arange(len(five_class)))
# ax.set_xticklabels(five_class, rotation=45)
plt.ylim((0, 1))
plt.ylabel("frequency")
plt.xlabel("size in mm")
plt.gca().set(title='size-frequency distribution (bramble leaves)', ylabel='relative frequency')

text = f"n = " + str(len(all_lengths_BRAMBLE))

plt.gca().text(0.8, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

print("Size-frequency distribution (median) plot produced for BRAMBLE n =", len(all_lengths_BRAMBLE),
      "valid tracks.")

plt.savefig(
    "Length-frequency distribution (median) plot ALL BRAMBLE.svg")

with open("ALL_LENGTHS_BRAMBLE.pickle", 'wb') as handle:
    pickle.dump(all_lengths_BRAMBLE, handle, protocol=pickle.HIGHEST_PROTOCOL)

# clear plot before moving on to rose data
plt.clf()

for video in all_rose:
    path = os.path.join(pose_absolute_path, video)
    print("Processing:", path)
    for r, d, f in os.walk(path):
        for file in f:
            df = pd.read_csv(os.path.join(path, file), delimiter=',', header=[0, 1, 2])

            x_diff = df["OmniTrax"]["b_t"]["x"].to_numpy() - df["OmniTrax"]["b_a_5"]["x"].to_numpy()
            y_diff = df["OmniTrax"]["b_t"]["y"].to_numpy() - df["OmniTrax"]["b_a_5"]["y"].to_numpy()
            lengths = np.sqrt(np.square(x_diff) + np.square(y_diff))
            median_length = np.round(np.median(lengths) / px_to_mm, 2)
            all_lengths_ROSE.append(median_length)

with open("ALL_LENGTHS_ROSE.pickle", 'wb') as handle:
    pickle.dump(all_lengths_ROSE, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plot Histogram on x
plt.hist(all_lengths_ROSE, bins=[2.5, 3, 3.5, 4, 4.5, 5], density=True, color=pink)
# ax.set_xticks(np.arange(len(five_class)))
# ax.set_xticklabels(five_class, rotation=45)
plt.ylim((0, 1))
plt.ylabel("frequency")
plt.xlabel("size in mm")
plt.gca().set(title='length-frequency distribution (rose petals)', ylabel='relative frequency')

text = f"n = " + str(len(all_lengths_ROSE))

plt.gca().text(0.8, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

print("Length-frequency distribution (median) plot produced for BRAMBLE n =", len(all_lengths_ROSE),
      "valid tracks.")

plt.savefig(
    "Length-frequency distribution (median) plot ALL ROSE.svg")

print("BRAMBLE MEAN:", np.round(np.mean(all_lengths_BRAMBLE), 2), "mm +/-",
      np.round(np.std(all_lengths_BRAMBLE), 2), "mm")

print("ROSE MEAN:", np.round(np.mean(all_lengths_ROSE), 2), "mm +/-",
      np.round(np.std(all_lengths_ROSE), 2), "mm")
