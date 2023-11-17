import pickle
import numpy as np
import matplotlib.pyplot as plt

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

with open("ALL_LENGTHS_BRAMBLE.pickle", 'wb') as handle:
    pickle.dump(all_lengths_BRAMBLE, handle, protocol=pickle.HIGHEST_PROTOCOL)

# clear plot before moving on to rose data
plt.clf()

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

print("Length-frequency distribution (median) plot produced for BRAMBLE n =", len(all_lengths_ROSE),
      "valid tracks.")

plt.savefig(
    "Length-frequency distribution (median) plot ALL ROSE.svg")

print("BRAMBLE MEAN:", np.round(np.mean(all_lengths_BRAMBLE), 2), "mm +/-",
      np.round(np.std(all_lengths_BRAMBLE), 2), "mm")

print("ROSE MEAN:", np.round(np.mean(all_lengths_ROSE), 2), "mm +/-",
      np.round(np.std(all_lengths_ROSE), 2), "mm")
