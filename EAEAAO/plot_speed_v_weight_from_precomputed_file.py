import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
from sklearn.linear_model import LinearRegression

input_file = "ALL_WEIGHTS_AND_SPEED_MEAN_IN_PX_per_S.pickle"

with open(input_file, 'rb') as pickle_file:
    weight_v_speed = pickle.load(pickle_file)

unique_classes = np.unique(weight_v_speed[:, 0])

sorted_weight_v_speed = [[] for i in range(len(unique_classes))]

# see I:\EAEAAO\CALIBRATION\2019-07-23_bramble_right\px_to_cm_conversion.xlsx

px_to_mm = 6.8628

#####

# simple regression analysis with loglog data

# get actual predicted weight back from classes
five_class = [1.3, 3.0, 6.8, 15.4, 35.1]

# lookup class
assigned_class = [five_class[int(i[0])] for i in weight_v_speed]

x = np.log10(np.array(assigned_class))
y = np.log10(weight_v_speed[:, 1])

model = LinearRegression()
model.fit(x.reshape((-1, 1)), y)
r_sq = model.score(x.reshape((-1, 1)), y)
y_pred = model.predict(x.reshape((-1, 1)))

print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}\n")
print(f"predicted response:\n{np.power(10, y_pred)}")

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"\ncoefficient of determination: {r_value}")
print(f"p-value: {p_value}")
print(f"intercept: {intercept}")
print(f"slope: {slope}\n")

plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots()
alpha = 1

ax.scatter(np.power(10, x),
           np.power(10, y),
           marker=None, cmap=None,
           vmin=0, vmax=10,
           alpha=alpha)

ax.plot(np.power(10, x), np.power(10, y_pred))

text = f"$R^2 = {r_sq:0.3f}$"

plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
               fontsize=14, verticalalignment='top')

ax.set_ylabel('mean speed [mm/s]')
ax.set_xlabel('body mass [mg]')
ax.set_title('mead speed over body mass [from classifier]')
ax.yaxis.grid(True)
ax.set_yscale('log')
ax.set_xscale('log')
# ax.set_ylim(1, 50)
# ax.set_xlim(2.5, 10)

# Save the figure and show
plt.tight_layout()

plt.savefig("body_mass_V_speed.svg", dpi='figure', pad_inches=0.1)

plt.show()

exit()

#####

total_mean_speed = np.mean(weight_v_speed[:, 1]) / px_to_mm
total_mean_speed_std = np.std(weight_v_speed[:, 1]) / px_to_mm

print("Average speed:", total_mean_speed, "+/-", total_mean_speed_std)

for elem in weight_v_speed:
    sorted_weight_v_speed[int(elem[0])].append(float(elem[1]) / px_to_mm)

for c, speed_list in enumerate(sorted_weight_v_speed):
    print(c, "contains", len(speed_list), "individuals")
    # check if each class is normally distributed to decide on statistical test
    print(stats.normaltest(speed_list))

kruskal = stats.kruskal(sorted_weight_v_speed[0],
                        sorted_weight_v_speed[1],
                        sorted_weight_v_speed[2],
                        sorted_weight_v_speed[3],
                        sorted_weight_v_speed[4])

print(kruskal)

dunn = sp.posthoc_dunn(sorted_weight_v_speed[:], p_adjust='bonferroni')
print(dunn)

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
