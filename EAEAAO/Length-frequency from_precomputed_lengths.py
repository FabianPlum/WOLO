import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats


def find_class(array, value):
    array_np = np.asarray(array)
    idx = (np.abs(array_np - value)).argmin()
    nearest_class = array_np[idx]
    pred_class = array.index(nearest_class)
    return pred_class


"""
classes = [0.0010, 0.0012, 0.0015, 0.0019, 0.0023, 0.0028,
           0.0034, 0.0042, 0.0052, 0.0064, 0.0078, 0.0096,
           0.0118, 0.0145, 0.0179, 0.0219, 0.0270, 0.0331,
           0.0407, 0.0500],
classes_upper_limit = [0.0011, 0.0013, 0.0017, 0.0021, 0.0025, 0.0031,
                       0.0038, 0.0047, 0.0058, 0.0071, 0.0087, 0.0107,
                       0.0131, 0.0162, 0.0199, 0.0244, 0.0300, 0.0369, 0.0453
                       ],
y_five_class_list = [0.0013, 0.0030, 0.0068, 0.0154, 0.0351],
y_five_class_list_upper_limit = [0.0021, 0.0047, 0.0107, 0.0244]
"""

bramble_file = "ALL_LENGTHS_BRAMBLE.pickle"
rose_file = "ALL_LENGTHS_ROSE.pickle"

with open(bramble_file, 'rb') as pickle_file:
    all_lengths_BRAMBLE = pickle.load(pickle_file)

with open(rose_file, 'rb') as pickle_file:
    all_lengths_ROSE = pickle.load(pickle_file)

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

bins = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

# Plot Histogram on x
plt.hist(all_lengths_BRAMBLE, bins=bins, density=True, color=green)
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

# clear plot before moving on to rose data
plt.clf()

###########

# now, do the same thing, but with mass estimated from body length relationships

body_mass = [1.10,
             1.30,
             1.50,
             1.90,
             2.40,
             2.70,
             3.70,
             4.20,
             5.30,
             7.00,
             7.80,
             10.10,
             12.50,
             14.20,
             19.70,
             23.50,
             29.20,
             34.70,
             39.10,
             47.10]

body_length = [2.71,
               2.90,
               2.94,
               3.04,
               3.32,
               3.44,
               3.61,
               3.87,
               4.23,
               4.51,
               4.96,
               5.46,
               5.53,
               5.85,
               6.81,
               7.02,
               7.26,
               7.96,
               8.32,
               8.60]

x = np.log10(body_length)
y = np.log10(body_mass)

model = LinearRegression()
model.fit(x.reshape((-1, 1)), y)
r_sq = model.score(x.reshape((-1, 1)), y)
y_pred = model.predict(x.reshape((-1, 1)))

print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}\n")
print(f"predicted response:\n{np.power(10, y_pred)}")

BRAMBLE_LEN_LOG = np.log10(np.array(all_lengths_BRAMBLE)).reshape((-1, 1))
BRAMBLE_WEIGHT = np.power(10, model.predict(BRAMBLE_LEN_LOG))

print("BRAMBLE WEIGHT MEAN:", np.round(np.mean(BRAMBLE_WEIGHT), 2), "mg +/-",
      np.round(np.std(BRAMBLE_WEIGHT), 2), "mg")

ROSE_LEN_LOG = np.log10(np.array(all_lengths_ROSE)).reshape((-1, 1))
ROSE_WEIGHT = np.power(10, model.predict(ROSE_LEN_LOG))

print("ROSE WEIGHT MEAN:", np.round(np.mean(ROSE_WEIGHT), 2), "mg +/-",
      np.round(np.std(ROSE_WEIGHT), 2), "mg")

# Plot Histogram on x
# five_class_bins = [0, 2.1, 4.7, 10.7, 24.4, 50]

five_class = [1.3, 3.0, 6.8, 15.4, 35.1]
five_class_limits = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

classes = [0.0010, 0.0012, 0.0015, 0.0019, 0.0023, 0.0028,
           0.0034, 0.0042, 0.0052, 0.0064, 0.0078, 0.0096,
           0.0118, 0.0145, 0.0179, 0.0219, 0.0270, 0.0331,
           0.0407, 0.0500]

classes = np.array(classes) * 1000

classes_upper_limit = [0.0011, 0.0013, 0.0017, 0.0021, 0.0025, 0.0031,
                       0.0038, 0.0047, 0.0058, 0.0071, 0.0087, 0.0107,
                       0.0131, 0.0162, 0.0199, 0.0244, 0.0300, 0.0369, 0.0453
                       ]
# hundred_class_limits = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

full_path = "I:/EAEAAO/RESULTS/weight_estimates_CLASS_MultiCamAnts-and-synth-simple_5_sigma-2_cross-entropy"

all_pred_classes = []

BRAMBLE_CLASSES = [find_class(classes.tolist(), float(x)) for x in BRAMBLE_WEIGHT]

ROSE_CLASSES = [find_class(classes.tolist(), float(x)) for x in ROSE_WEIGHT]

plt.hist(BRAMBLE_CLASSES, bins=np.arange(0, 20).tolist(), density=True, color=green)
# ax.set_xticks(np.arange(len(five_class)))
# ax.set_xticklabels(five_class, rotation=45)
plt.ylim((0, 1))
plt.ylabel("frequency")
plt.xlabel("weight mg")
plt.gca().set(title='size-frequency distribution (bramble leaves)', ylabel='relative frequency')

text = f"n = " + str(len(all_lengths_BRAMBLE))

plt.gca().text(0.8, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

print("Size-frequency distribution (median) plot produced for BRAMBLE n =", len(all_lengths_BRAMBLE),
      "valid tracks.")

plt.savefig("Size-frequency distribution from pose (median) plot ALL BRAMBLE - ULTRA-FINE.svg")

# clear plot before moving on to rose data
plt.clf()

###########

plt.hist(ROSE_CLASSES, bins=np.arange(0, 20).tolist(), density=True, color=pink)
# ax.set_xticks(np.arange(len(five_class)))
# ax.set_xticklabels(five_class, rotation=45)
plt.ylim((0, 1))
plt.ylabel("frequency")
plt.xlabel("weight mg")
plt.gca().set(title='size-frequency distribution (rose petals)', ylabel='relative frequency')

text = f"n = " + str(len(all_lengths_ROSE))

plt.gca().text(0.8, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

print("Size-frequency distribution (median) plot produced for ROSE n =", len(all_lengths_ROSE),
      "valid tracks.")

plt.savefig("Size-frequency distribution from pose (median) plot ALL ROSE - ULTRA-FINE.svg")

# clear plot before moving on to rose data
plt.clf()

##########

# Plot Histogram on x
plt.hist(all_lengths_ROSE, bins=bins, density=True, color=pink)
# ax.set_xticks(np.arange(len(five_class)))
# ax.set_xticklabels(five_class, rotation=45)
plt.ylim((0, 1))
plt.ylabel("frequency")
plt.xlabel("size in mm")
plt.gca().set(title='length-frequency distribution (rose petals)', ylabel='relative frequency')

text = f"n = " + str(len(all_lengths_ROSE))

plt.gca().text(0.8, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

print("Length-frequency distribution (median) plot produced for ROSE n =", len(all_lengths_ROSE),
      "valid tracks.")

plt.savefig(
    "Length-frequency distribution (median) plot ALL ROSE.svg")

print("BRAMBLE MEAN:", np.round(np.mean(all_lengths_BRAMBLE), 2), "mm +/-",
      np.round(np.std(all_lengths_BRAMBLE), 2), "mm")

print("ROSE MEAN:", np.round(np.mean(all_lengths_ROSE), 2), "mm +/-",
      np.round(np.std(all_lengths_ROSE), 2), "mm")

########

# Finally, run some basic stats on the extracted distributions

print("bramble normal distribution check:", stats.normaltest(all_lengths_BRAMBLE))
print("rose normal distribution check:", stats.normaltest(all_lengths_ROSE))

kruskal = stats.kruskal(all_lengths_BRAMBLE,
                        all_lengths_ROSE)
print("Kruskal - bramble v rose")
print(kruskal)
