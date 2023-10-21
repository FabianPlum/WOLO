"""
WOLO - compute regression and classification metrics
This script takes the test_data_pred_results.csv files produced during network fitting and evaluation on test-data as an input and computes the desired output metrics and plots:

MAPE_true
MAPE_ideal
MAPE_class
classification accuracy
confusion matrices
class-wise scores
coefficient of variation
"""
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn import metrics


def goAPE(y_true, y_pred, CLASS_LIST=None, gt_v_class=False):
    """
    y_true : gt label vector of lenght n
    y_pred : prediction label vector of length n
    CLASS_LIST : lookup table of class centres (optional)
    return : MAPE, STDAPE
    """
    assert len(y_true) == len(y_pred), "Mismatch between input vectors"
    if CLASS_LIST is None:
        APE = [np.abs((x[0] - x[1]) / x[0]) for x in zip(y_true, y_pred) if x[1] != -1]
    elif gt_v_class:
        APE = [np.abs((x[0] - CLASS_LIST[x[1]]) / x[0]) for x in zip(y_true, y_pred) if x[1] != -1]
    else:
        APE = [np.abs((CLASS_LIST[x[0]] - CLASS_LIST[x[1]]) / CLASS_LIST[x[0]]) for x in zip(y_true, y_pred) if
               x[1] != -1]

    MAPE = 100 * np.mean(APE)
    STDAPE = 100 * np.std(APE)

    return MAPE, STDAPE


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=str)
    ap.add_argument("-o", "--output", required=False, default="", type=str)
    ap.add_argument("-v", "--verbose", required=False, default=False, type=bool)
    ap.add_argument("-m", "--known_ID", required=False, default=False, type=bool)

    args = vars(ap.parse_args())

    OUTPUT_LOCATION = args["output"]
    input_folder = args["input"]

    # call the following once to produce resized plots across the notebook
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['figure.dpi'] = 100

    input_file = input_folder.replace("\\", "/") + "/test_data_pred_results.csv"
    output_name = input_file.split("/")[-2]

    print(output_name)
    output_txt = open(os.path.join(OUTPUT_LOCATION, output_name + "---ALL_OUTPUTS.txt"), "w")

    output_txt.write("Running evaluation of inference outputs produced by: " + output_name + " ...\n")
    print("Beginning writing to output file...")
    file = open(input_file, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()

    if input_file.split("/")[-1].split("_")[0] == "DETECT":
        DETECTION = True
    else:
        DETECTION = False


    def find_class(array, value):
        array_np = np.asarray(array)
        idx = (np.abs(array_np - value)).argmin()
        nearest_class = array_np[idx]
        pred_class = array.index(nearest_class)
        return pred_class


    print("Retrieving the following info from the input file:", data[0][1:6])
    file_names = [row[1] for row in data[1:]]

    five_class = [0.0013, 0.0030, 0.0068, 0.0154, 0.0351]
    twenty_class = [0.0010, 0.0012, 0.0015, 0.0019, 0.0023, 0.0028,
                    0.0034, 0.0042, 0.0052, 0.0064, 0.0078, 0.0096,
                    0.0118, 0.0145, 0.0179, 0.0219, 0.0270, 0.0331,
                    0.0407, 0.0500]
    scaled_20 = [int(x * 10001) for x in twenty_class]

    if len(data[0]) > 11:
        CLASS_LIST = twenty_class
    elif len(data[0]) == 11:
        CLASS_LIST = five_class
    else:
        CLASS_LIST = twenty_class  # use classification approach of 20 class list for displaying regressor outputs

    if len(data[0]) < 6 and not DETECTION:  # regressors have fewer lines as the output activations aren't relevant
        true_classes = [scaled_20.index(int(x.replace("\\", "/").split("/")[-2])) for x in [row[1] for row in data[1:]]]
        pred_classes = [find_class(twenty_class, float(x)) for x in [row[3] for row in data[1:]]]
        true_weight = [float(x) for x in [row[2] for row in data[1:]]]
        pred_weight = [float(x) for x in [row[3] for row in data[1:]]]
    else:
        true_classes = [int(x) for x in [row[2] for row in data[1:]]]
        pred_classes = [int(x) for x in [row[3] for row in data[1:]]]
        true_weight = [float(x) for x in [row[4] for row in data[1:]]]
        pred_weight = [float(x) for x in [row[5] for row in data[1:]]]

    if DETECTION:
        if len(np.unique(true_classes)) == 5:
            CLASS_LIST = five_class
        else:
            CLASS_LIST = twenty_class

        msg = str(len(np.unique(true_classes))) + " class detection data found!"
        print(msg)
        output_txt.write(msg + "\n\n")

    """    
    Compute overall MAPE and accuracy scores across the full test dataset
    """

    MAPE_true, STDAPE_true = goAPE(y_true=true_weight,
                                   y_pred=pred_weight)
    msg = "MAPE_true  : " + str(round(MAPE_true, 2)) + "  STDAPE_true : " + str(round(STDAPE_true, 2))
    output_txt.write(msg + "\n")
    print(msg)

    if CLASS_LIST is not None:
        MAPE_ideal, STDAPE_ideal = goAPE(y_true=true_weight,
                                         y_pred=true_classes,
                                         CLASS_LIST=CLASS_LIST,
                                         gt_v_class=True)
        msg = "MAPE_ideal : " + str(round(MAPE_ideal, 2)) + "  STDAPE_ideal : " + str(round(STDAPE_ideal, 2))
        output_txt.write(msg + "\n")
        print(msg)

        MAPE_class, STDAPE_class = goAPE(y_true=true_classes,
                                         y_pred=pred_classes,
                                         CLASS_LIST=CLASS_LIST)
        msg = "MAPE_class : " + str(round(MAPE_class, 2)) + "  STDAPE_class : " + str(round(STDAPE_class, 2))
        output_txt.write(msg + "\n")
        print(msg)

        accuracy = metrics.accuracy_score(y_true=true_classes,
                                          y_pred=pred_classes)

        msg = "Classification accuracy : " + str(round(accuracy, 4))
        output_txt.write(msg + "\n\n")
        print(msg)

    """    
    Produce confusion matrices
    """

    if DETECTION:
        true_classes_cleaned = [x[0] for x in zip(true_classes, pred_classes) if x[1] != -1]
        pred_classes_cleaned = [x for x in pred_classes if x != -1]
        file_names_cleaned = [x[0] for x in zip(file_names, pred_classes) if x[1] != -1]
        true_weight_cleaned = [x[0] for x in zip(true_weight, pred_classes) if x[1] != -1]
        pred_weight_cleaned = [x[0] for x in zip(pred_weight, pred_classes) if x[1] != -1]

        true_classes = true_classes_cleaned
        pred_classes = pred_classes_cleaned
        file_names = file_names_cleaned
        true_weight = true_weight_cleaned
        pred_weight = pred_weight_cleaned

    y_actu = pd.Series([CLASS_LIST[x] for x in true_classes], name='True class')
    y_pred = pd.Series([CLASS_LIST[x] for x in pred_classes], name='Predicted class')
    df_confusion = pd.crosstab(y_actu, y_pred)
    df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis="index")

    df_conf_norm.to_csv(os.path.join(OUTPUT_LOCATION, output_name + "---Confusion_matrix.csv"))

    confusion_matrix = metrics.confusion_matrix(true_classes, pred_classes, normalize="true")
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=CLASS_LIST)

    cm_display.plot(cmap=plt.cm.gray_r,
                    include_values=False,  # set to true to display individual values
                    xticks_rotation=45)

    # update colour scale to enforce displaying normalised values between 0 and 1
    for im in plt.gca().get_images():
        im.set_clim(vmin=0, vmax=1)

    plt.title("Confusion matrix")
    plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---Confusion_matrix.svg"), dpi='figure', pad_inches=0.1)
    if args["verbose"]:
        plt.show()

    """
    Compute class wise scores
    and prediction stability in terms of the average coefficient of variation across continuous samples containing the same individual
    """

    data_comb = zip(file_names, true_classes, pred_classes, true_weight, pred_weight)

    prev_class_temp = true_classes[0]
    ind_list = {}

    # we need to enforce equal lengths of CLASS_LIST & class_wise_scores
    class_wise_scores = [[] for i in range(len(CLASS_LIST))]

    class_wise_elements_gt_cl = []
    class_wise_elements_p_cl = []
    class_wise_elements_gt = []
    class_wise_elements_p = []

    output_txt.write("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
    output_txt.write("\n Class-wise scores: \n")

    for f, gt_cl, p_cl, gt, p in data_comb:
        file_components = f.split("/")
        class_temp = gt_cl
        # cut away the frame number so individuals have consistent names
        vid = "_".join(file_components[2].split("_")[0:-2]) + "_" + file_components[2].split("_")[-1]
        # alternatively, if no consistent identities are given in the video path, assume
        # a change in gt weight corresponds to a new individual
        if not args["known_ID"]:
            if str(gt) not in ind_list:
                ind_list[str(gt)] = []

        else:
            if vid not in ind_list:
                ind_list[vid] = []

        """
        # use the following instead of line below, when extracting error stability instead of prediction stability
        if CLASS_LIST is not None:
            APE_temp = np.abs((CLASS_LIST[gt_cl] - CLASS_LIST[p_cl])/CLASS_LIST[gt_cl])
        else:
            APE_temp = np.abs((gt - p)/gt)
            
        ind_list[vid].append([gt, APE_temp])
        """

        if not args["known_ID"]:
            ind_list[str(gt)].append([gt, p])
        else:
            ind_list[vid].append([gt, p])

        if class_temp != prev_class_temp or f == file_names[-1]:
            if f == file_names[-1]:
                # in case this is the last element, add the final line before computing scores
                class_wise_elements_gt_cl.append(gt_cl)
                class_wise_elements_p_cl.append(p_cl)
                class_wise_elements_gt.append(gt)
                class_wise_elements_p.append(p)

            msg = "\nCLASS : " + str(CLASS_LIST[prev_class_temp])
            output_txt.write(msg + "\n")
            print(msg)

            MAPE_true, STDAPE_true = goAPE(y_true=class_wise_elements_gt,
                                           y_pred=class_wise_elements_p)
            msg = "MAPE_true  : " + str(round(MAPE_true, 2)) + "  STDAPE_true : " + str(round(STDAPE_true, 2))
            output_txt.write(msg + "\n")
            print(msg)

            MAPE_ideal, STDAPE_ideal = goAPE(y_true=class_wise_elements_gt,
                                             y_pred=class_wise_elements_gt_cl,
                                             CLASS_LIST=CLASS_LIST,
                                             gt_v_class=True)
            msg = "MAPE_ideal : " + str(round(MAPE_ideal, 2)) + "  STDAPE_ideal : " + str(round(STDAPE_ideal, 2))
            output_txt.write(msg + "\n")
            print(msg)

            MAPE_class, STDAPE_class = goAPE(y_true=class_wise_elements_gt_cl,
                                             y_pred=class_wise_elements_p_cl,
                                             CLASS_LIST=CLASS_LIST)
            msg = "MAPE_class : " + str(round(MAPE_class, 2)) + "  STDAPE_class : " + str(round(STDAPE_class, 2))
            output_txt.write(msg + "\n")
            print(msg)

            accuracy = metrics.accuracy_score(y_true=class_wise_elements_gt_cl,
                                              y_pred=class_wise_elements_p_cl)
            msg = "Classification accuracy : " + str(round(accuracy, 4))
            output_txt.write(msg + "\n")
            print(msg)

            class_wise_scores[class_temp] = [prev_class_temp,
                                             MAPE_true, STDAPE_true,
                                             MAPE_ideal, STDAPE_ideal,
                                             MAPE_class, STDAPE_class,
                                             accuracy]

            prev_class_temp = class_temp
            class_wise_elements_gt_cl = []
            class_wise_elements_p_cl = []
            class_wise_elements_gt = []
            class_wise_elements_p = []

        class_wise_elements_gt_cl.append(gt_cl)
        class_wise_elements_p_cl.append(p_cl)
        class_wise_elements_gt.append(gt)
        class_wise_elements_p.append(p)

    # clean class_wise_scores, in case not all classes are represented
    class_wise_scores = [[i, 0, 0, 0, 0, 0, 0, 0] if elem == [] else elem for i, elem in enumerate(class_wise_scores)]

    class_wise_scores = np.array(class_wise_scores)

    coeff_var_ind = []

    for key, value in ind_list.items():
        coeff_var_ind.append(np.std([i[1] for i in value]) / np.mean([i[1] for i in value]))

    msg = "Average coefficient of variation across repeated predictions: " + str(round(np.mean(coeff_var_ind), 4))

    print(msg)
    output_txt.write("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
    output_txt.write("\n" + msg + "\n")
    output_txt.close()

    """
    class wise score visualisation
    Finally, plot the resulting class-wise MAPE (comparing prediction to ground truth, regardless of inference method) and class-wise accuracy
    """

    plt.rcParams['figure.figsize'] = [6, 4]
    plt.rcParams['figure.dpi'] = 100
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(CLASS_LIST)), class_wise_scores[:, 1],
           # yerr=class_wise_scores[:,2],
           align='center',
           alpha=0.5,
           ecolor='black', capsize=10)

    ax.set_ylabel('MAPE')
    ax.set_xticks(np.arange(len(CLASS_LIST)))
    ax.set_xticklabels(CLASS_LIST, rotation=45)
    ax.set_title('class-wise MAPE')
    ax.yaxis.grid(True)
    # ax.set_yscale('log')
    ax.set_ylim(1, 500)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---class-wise_MAPE.svg"), dpi='figure', pad_inches=0.1)
    if args["verbose"]:
        plt.show()
    plt.rcParams['figure.figsize'] = [6, 4]
    plt.rcParams['figure.dpi'] = 100
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(CLASS_LIST)), class_wise_scores[:, -1],
           # yerr=class_wise_scores[:,2],
           align='center',
           alpha=0.5,
           ecolor='black', capsize=10)

    ax.set_ylabel('MAPE')
    ax.set_xticks(np.arange(len(CLASS_LIST)))
    ax.set_xticklabels(CLASS_LIST, rotation=45)
    ax.set_title('class-wise accuracy')
    ax.yaxis.grid(True)
    ax.set_ylim(0, 1)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---class-wise_accuracy.svg"), dpi='figure', pad_inches=0.1)
    if args["verbose"]:
        plt.show()

    # Plot ground truth vs predicted weights (log-log-scaled)
    gt_v_pred_xy = []

    for key, value in ind_list.items():
        gt_v_pred_xy.append([value[0][0], np.mean([i[1] for i in value])])

    plt.rcParams['figure.figsize'] = [6, 6]
    plt.rcParams['figure.dpi'] = 100
    fig, ax = plt.subplots()
    if args["known_ID"]:
        alpha = 0.1
    else:
        alpha = 0.4
    ax.scatter([i[0] for i in gt_v_pred_xy],
               [i[1] for i in gt_v_pred_xy],
               marker=None, cmap=None,
               vmin=0.0005, vmax=0.05,
               alpha=alpha)

    ax.plot([0.0005, 0.05], [0.0005, 0.05], linewidth=1.0)

    """
    ax.set_xticks(CLASS_LIST)
    ax.set_xticklabels(CLASS_LIST, rotation=45)

    ax.set_yticks(CLASS_LIST)
    ax.set_yticklabels(CLASS_LIST)
    """

    ax.set_ylabel('predicted weight [g]')
    ax.set_xlabel('ground truth weight [g]')
    ax.set_title('gt vs predicted weight')
    ax.yaxis.grid(True)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(0.001, 0.05)
    ax.set_xlim(0.001, 0.05)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---gt_vs_predicted_weight.svg"),
                dpi='figure', pad_inches=0.1)

    if args["verbose"]:
        plt.show()
