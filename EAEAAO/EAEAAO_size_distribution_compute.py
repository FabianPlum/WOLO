import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats


def find_class(array, value):
    array_np = np.asarray(array)
    idx = (np.abs(array_np - value)).argmin()
    nearest_class = array_np[idx]
    pred_class = array.index(nearest_class)
    return pred_class


def clean_array(array, strip_NaN=True, strip_zero=False):
    array_np = np.asarray(array)
    if strip_NaN:
        array_np = array_np[np.logical_not(np.isnan(array_np))]
    if strip_zero:
        array_np = array_np[np.nonzero(array_np)]
    return array_np


all_bramble = ["2019-07-22_bramble_left_ALL_WEIGHTS.pickle",
               "2019-07-23_bramble_right2_ALL_WEIGHTS.pickle",
               "2019-07-23_bramble_right_ALL_WEIGHTS.pickle",
               "2019-07-24_bramble_left_ALL_WEIGHTS.pickle",
               "2019-07-24_bramble_right_ALL_WEIGHTS.pickle",
               "2019-07-30_bramble_left_ALL_WEIGHTS.pickle",
               "2019-07-31_bramble_left_ALL_WEIGHTS.pickle",
               "2019-07-31_bramble_right_ALL_WEIGHTS.pickle",
               "2019-08-01_bramble_left_ALL_WEIGHTS.pickle",
               "2019-08-03_bramble-left_ALL_WEIGHTS.pickle",
               "2019-08-03_bramble-right_ALL_WEIGHTS.pickle",
               "2019-08-05_bramble_left_ALL_WEIGHTS.pickle",
               "2019-08-06_bramble_left_ALL_WEIGHTS.pickle",
               "2019-08-07_bramble_left_ALL_WEIGHTS.pickle",
               "2019-08-07_bramble_right_ALL_WEIGHTS.pickle",
               "2019-08-09_bramble_left_ALL_WEIGHTS.pickle",
               "2019-08-13_bramble_right_ALL_WEIGHTS.pickle",
               "2019-08-15_bramble_left_ALL_WEIGHTS.pickle",
               "2019-08-15_bramble_right_ALL_WEIGHTS.pickle",
               "2019-08-16_bramble_right_ALL_WEIGHTS.pickle",
               "2019-08-22_bramble_right_ALL_WEIGHTS.pickle"]

all_rose = ["2019-07-23_rose_left_2_ALL_WEIGHTS.pickle",
            "2019-07-23_rose_left_ALL_WEIGHTS.pickle",
            "2019-07-25_rose_left_ALL_WEIGHTS.pickle",
            "2019-07-25_rose_right_ALL_WEIGHTS.pickle",
            "2019-07-30_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-01_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-05_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-06_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-08_rose_left_ALL_WEIGHTS.pickle",
            "2019-08-08_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-09_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-12_rose_left_ALL_WEIGHTS.pickle",
            "2019-08-12_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-13_rose_left_ALL_WEIGHTS.pickle",
            "2019-08-16_rose_left_ALL_WEIGHTS.pickle",
            "2019-08-20_rose_left_ALL_WEIGHTS.pickle",
            "2019-08-20_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-21_rose_left_ALL_WEIGHTS.pickle",
            "2019-08-21_rose_right_ALL_WEIGHTS.pickle",
            "2019-08-22_rose_left_ALL_WEIGHTS.pickle"]

five_class = [0.0013, 0.0030, 0.0068, 0.0154, 0.0351]
five_class_limits = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

full_path = "I:/EAEAAO/RESULTS/weight_estimates_CLASS_MultiCamAnts-and-synth-simple_5_sigma-2_cross-entropy"

all_pred_classes = []

for input_file in all_bramble:
    with open(os.path.join(full_path, input_file), 'rb') as pickle_file:
        all_weights_temp = pickle.load(pickle_file)
        all_weights_compressed_median = np.round(np.nanmedian(all_weights_temp, axis=0), 4)

        # produced using MEDIAN
        pred_classes = [find_class(five_class, float(x)) for x in all_weights_compressed_median]
        pred_classes = clean_array(pred_classes, strip_NaN=True)

    all_pred_classes.extend(pred_classes)

"""
plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

color = (0.2,  # redness
         0.4,  # greenness
         0.2,  # blueness
         0.6  # transparency
         )

# Plot Histogram on x
fig, ax = plt.subplots()
ax.hist(pred_classes, bins=five_class_limits, density=True, color=color)
ax.set_xticks(np.arange(len(five_class)))
ax.set_xticklabels(five_class, rotation=45)
ax.set_ylim(0, 1)
plt.gca().set(title='size-frequency distribution (bramble leaves)', ylabel='relative frequency')

text = f"n = " + str(len(all_pred_classes))

plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

print("Size-frequency distribution (median) plot produced for BRAMBLE n =", len(all_pred_classes),
      "valid tracks.")

plt.savefig(
    "Size-frequency distribution (median) plot ALL BRAMBLE median.svg")

"""

bramble = all_pred_classes

all_pred_classes = []

for input_file in all_rose:
    with open(os.path.join(full_path, input_file), 'rb') as pickle_file:
        all_weights_temp = pickle.load(pickle_file)
        all_weights_compressed_median = np.round(np.nanmedian(all_weights_temp, axis=0), 4)

        # produced using MEDIAN
        pred_classes = [find_class(five_class, float(x)) for x in all_weights_compressed_median]
        pred_classes = clean_array(pred_classes, strip_NaN=True)

    all_pred_classes.extend(pred_classes)

"""
plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

color = (0.9,  # redness
         0,  # greenness
         0.2,  # blueness
         0.6  # transparency
         )

# Plot Histogram on x
fig, ax = plt.subplots()
ax.hist(pred_classes, bins=five_class_limits, density=True, color=color)
ax.set_xticks(np.arange(len(five_class)))
ax.set_xticklabels(five_class, rotation=45)
ax.set_ylim(0, 1)
plt.gca().set(title='size-frequency distribution (rose petals)', ylabel='relative frequency')

text = f"n = " + str(len(all_pred_classes))

plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

print("Size-frequency distribution (median) plot produced for ROSE n =", len(all_pred_classes),
      "valid tracks.")

plt.savefig(
    "Size-frequency distribution (median) plot ALL ROSE median.svg")

"""

rose = all_pred_classes

print("bramble normal distribution check:", stats.normaltest(bramble))
print("rose normal distribution check:", stats.normaltest(rose))

kruskal = stats.kruskal(bramble,
                        rose)
print("Kruskal - bramble v rose")
print(kruskal)

print("\nrose size frequency:")
for c in np.unique(np.array(rose)):
    print(np.count_nonzero(rose == c))

print("\nbramble size frequency:")
for c in np.unique(np.array(bramble)):
    print(np.count_nonzero(bramble == c))

# additionally compare pose-derived body-length to direct size inference
bramble_file = "ALL_LENGTHS_BRAMBLE.pickle"

classes_bodylength = [3, 4, 5, 6, 7]

with open(bramble_file, 'rb') as pickle_file:
    all_lengths_BRAMBLE = pickle.load(pickle_file)

bramble_pose = []

print("getting classes from body-lengths...")
print(all_lengths_BRAMBLE[0:20])
for elem in all_lengths_BRAMBLE:
    bramble_pose.append(find_class(classes_bodylength, elem))

print("\nbramble POSE size frequency:")
for c in np.unique(np.array(bramble_pose)):
    print(np.count_nonzero(bramble_pose == c))

print("\nbramble-pose normal distribution check:", stats.normaltest(bramble_pose))

kruskal_pose = stats.kruskal(bramble,
                             bramble_pose)

print("Kruskal - pose v direct inference - bramble")
print(kruskal_pose)

"""
all_pred_classes = []

all_bramble.extend(all_rose)

for input_file in all_bramble:
    with open(os.path.join(full_path,input_file), 'rb') as pickle_file:
        all_weights_temp = pickle.load(pickle_file)
        all_weights_compressed_median = np.round(np.nanmedian(all_weights_temp, axis=0), 4)

        # produced using MEDIAN
        pred_classes = [find_class(five_class, float(x)) for x in all_weights_compressed_median]
        pred_classes = clean_array(pred_classes, strip_NaN=True)

    all_pred_classes.extend(pred_classes)

plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

color = (0.2,  # redness
         0.2,  # greenness
         0.7,  # blueness
         0.6  # transparency
         )

# Plot Histogram on x
fig, ax = plt.subplots()
ax.hist(pred_classes, bins=five_class_limits, density=True, color=color)
ax.set_xticks(np.arange(len(five_class)))
ax.set_xticklabels(five_class, rotation=45)
ax.set_ylim(0, 1)
plt.gca().set(title='size-frequency distribution (all animals)', ylabel='relative frequency')

text = f"n = " + str(len(all_pred_classes))

plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

print("Size-frequency distribution (median) plot produced for all animals n =", len(all_pred_classes),
      "valid tracks.")

plt.savefig(
    "Size-frequency distribution (median) plot ALL COMBINED median.svg")
"""
