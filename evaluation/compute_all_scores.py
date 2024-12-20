"""
WOLO - compute regression and classification metrics
This script takes the test_data_pred_results.csv files produced during network fitting and evaluation on test-data as
an input and computes the desired output metrics and plots:

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
import scipy.stats as sp


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


def find_class(array, value):
    array_np = np.asarray(array)
    idx = (np.abs(array_np - value)).argmin()
    nearest_class = array_np[idx]
    pred_class = array.index(nearest_class)
    return pred_class


def compute_scores(input_file,
                   output,
                   verbose=False,
                   known_ID=False,
                   gt_from_name=False,
                   detection_format=False,
                   dataset_name=None,
                   create_plots=True,
                   exclude_aug=False):
    """
    replaces original main method, runs all analysis computing
    """

    results_dict = {
        "model": os.path.basename(input_file).split("---")[0],
        "dataset": dataset_name,
        "inference_type": os.path.basename(input_file).split("_")[0],
        "real data": None,
        "synth data": None,
        "sigma": None,
        "R^2 linear": None,
        "R^2 log": None,
        "classification accuracy": None,
        "MAPE_true": None,
        "MAPE_std": None,
        "MAPE_ideal": None,
        "COV": None,
        "Prediction_Stability": None,
        "Spearman rank-order": None,
        "Spearman rank-order p-value": None,
        "Absolute Accuracy bias": None,
        "Absolute Accuracy bias p-value": None,
        "Relative Accuracy bias": None,
        "Relative Accuracy bias p-value": None,
        "Precision bias": None,
        "Precision bias p-value": None
    }

    """
    get all info about the model type and its training data
    """
    # exclude "aug" models (due to inconsistent results with custom augmentation scripts)
    if exclude_aug:
        # CLASS_MultiCamAnts-and-synth-simple_20_sigma-2_cross-entropy_aug--
        if results_dict["model"].split("---")[0].split("_")[-1] == "aug":
            return

    # real data
    if os.path.basename(input_file).split("_")[1].split("-")[0] == "MultiCamAnts":
        results_dict["real data"] = "MultiCamAnts"

    # synth data
    if any([True if i == "synth" else False for i in os.path.basename(input_file).split("_")[1].split("-")]):
        results_dict["synth data"] = os.path.basename(input_file).split("_")[1].split("synth-")[-1]

    # label smoothing
    if results_dict["inference_type"] == "CLASS":
        if os.path.basename(input_file).split("_")[3].split("-")[0] == "sigma":
            results_dict["sigma"] = float(os.path.basename(input_file).split("_")[3].split("-")[-1])

    # refine
    if "REFINE" in os.path.basename(input_file):
        results_dict["refine"] = "yes"
    else:
        results_dict["refine"] = "no"

    OUTPUT_LOCATION = output
    input_folder = input_file

    if dataset_name is not None:
        print("INFO: Producing results for", input_file, "with dataset", dataset_name)

    if create_plots:
        # call the following once to produce resized plots across the notebook
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.rcParams['figure.dpi'] = 100

    if detection_format:
        DETECTION = True
        input_file = input_folder.replace("\\", "/")
        if input_file[-4:] != ".csv":
            return  # skip pkl files
        else:
            file_specifics = input_file.split("/")[-1].split("test_data")[0]
    else:
        DETECTION = False
        input_file = input_folder.replace("\\", "/") + "/test_data_pred_results.csv"
        file_specifics = ""

    file = open(input_file, "r")
    try:
        data = list(csv.reader(file, delimiter=","))
        file.close()
    except UnicodeDecodeError:
        file.close()
        return

    if len(data) < 20:
        # ignore short files which do not contain correctly formatted data
        return

    output_name = file_specifics + input_file.split("/")[-2]

    print(output_name)
    output_txt = open(os.path.join(OUTPUT_LOCATION, output_name + "---ALL_OUTPUTS.txt"), "w")

    output_txt.write("Running evaluation of inference outputs produced by: " + output_name + " ...\n")
    print("Beginning writing to output file...")

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
        REGRESSION = False
    elif len(data[0]) == 11:
        CLASS_LIST = five_class
        REGRESSION = False
    else:
        CLASS_LIST = twenty_class  # use classification approach of 20 class list for displaying regressor outputs
        if DETECTION:
            REGRESSION = False
        else:
            REGRESSION = True

    if REGRESSION:  # regressors have fewer lines as the output activations aren't relevant
        true_classes = [min(range(len(scaled_20)), key=lambda i: abs(scaled_20[i] - int(x.replace("\\", "/").split("/")[-2]))) for x in [row[1] for row in data[1:]]]
        pred_classes = [find_class(twenty_class, float(x)) for x in [row[3] for row in data[1:]]]
        true_weight = [float(x) for x in [row[2] for row in data[1:]]]
        pred_weight = [float(x) for x in [row[3] for row in data[1:]]]
    else:
        try:
            true_classes = [int(x) for x in [row[2] for row in data[1:]]]
        except ValueError:
            print("\n\n---------------\nIncorrect file type! Skipping file...\n---------------\n\n")
            return
        pred_classes = [int(x) for x in [row[3] for row in data[1:]]]
        if gt_from_name:
            true_weight = [float(x.split("-")[-1][:-4]) / 10000 for x in [row[1] for row in data[1:]]]
        else:
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
        if REGRESSION:
            MAPE_ideal = 0
            STDAPE_ideal = 0
        else:
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
    Update results dict with MAPEs & accuracy
    """

    results_dict["MAPE_true"] = MAPE_true  # overwritten below -> report MAPE
    results_dict["MAPE_ideal"] = MAPE_ideal
    results_dict["classification accuracy"] = accuracy

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

    if create_plots:
        df_confusion = pd.crosstab(y_actu, y_pred)
        df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis="index")

        df_conf_norm.to_csv(os.path.join(OUTPUT_LOCATION, output_name + "---Confusion_matrix.csv"))

        confusion_matrix = metrics.confusion_matrix(true_classes, pred_classes, normalize="true",
                                                    labels=range(0, len(CLASS_LIST)))

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                    display_labels=CLASS_LIST)

        cm_display.plot(cmap=plt.cm.gray_r,
                        include_values=False,  # set to true to display individual values
                        xticks_rotation=45)

        # update colour scale to enforce displaying normalised values between 0 and 1
        for im in plt.gca().get_images():
            im.set_clim(vmin=0, vmax=1)

        plt.title("Confusion matrix")
        plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---Confusion_matrix.svg"), dpi='figure',
                    pad_inches=0.1)
        if verbose:
            plt.show()
        plt.close()

    """
    Compute class wise scores
    and prediction stability in terms of the average coefficient of variation across continuous samples containing the 
    same individual
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
        if os.name == 'nt':  # for Windows
            file_components = f.split("\\")
        else:  # for Linux
            file_components = f.split("/")
        class_temp = gt_cl
        # cut away the frame number so individuals have consistent names
        vid = "_".join(file_components[2].split("_")[0:-2]) + "_" + file_components[2].split("_")[-1]

        # alternatively, if no consistent identities are given in the video path, assume
        # a change in gt weight corresponds to a new individual
        # in the validation split of MultiCamAnts we need to add a new ID for individuals with the EXACT same weight

        # okay, this is going to feel hacky, but I don't want to rename all the data.
        # SO. let's filter explicitly to correctly group repeated predictions.
        # the first TWO words + GT are the key EXCEPT for PLAIN

        if not known_ID:
            vid_parts = vid.split("_")
            if vid_parts[0] == "PLAIN":
                if vid_parts[1] == "FRAGMENTS" or vid_parts[1] == "LEAF":
                    id_key = "PLAIN_FRAGMENTS-" + str(gt)
                else:
                    id_key = "PLAIN-" + str(gt)
            else:
                id_key = '_'.join(vid_parts[0:2]) + "-" + str(gt)

            if id_key not in ind_list:
                ind_list[id_key] = []

            ind_list[id_key].append([gt, p])

        else:
            if vid not in ind_list:
                ind_list[vid] = []
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

            class_wise_scores[prev_class_temp] = [prev_class_temp,
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

    cov = round(np.mean(coeff_var_ind), 4)
    msg = "Average coefficient of variation across repeated predictions: " + str(cov)

    results_dict["COV"] = cov

    print(msg)
    output_txt.write("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
    output_txt.write("\n" + msg + "\n")

    """
    class wise score visualisation
    Finally, plot the resulting class-wise MAPE (comparing prediction to ground truth, regardless of inference method) 
    and class-wise accuracy
    """

    if create_plots:
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['figure.dpi'] = 100
        fig, ax = plt.subplots()

        ax.scatter(np.arange(len(CLASS_LIST)), class_wise_scores[:, -1])
        # alt: bar plots
        """
        ax.bar(np.arange(len(CLASS_LIST)), class_wise_scores[:, -1],
               # yerr=class_wise_scores[:,2],
               align='center',
               alpha=0.5,
               ecolor='black', capsize=10)
        """

        ax.set_ylabel('categorical accuracy')
        ax.set_xticks(np.arange(len(CLASS_LIST)))
        ax.set_xticklabels(CLASS_LIST, rotation=45)
        ax.set_title('class-wise accuracy')
        ax.yaxis.grid(True)
        ax.set_ylim(0, 1)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---class-wise_accuracy.svg"), dpi='figure',
                    pad_inches=0.1)
        if verbose:
            plt.show()
        plt.close()

    # Plot ground truth vs predicted weights (log-log-scaled)
    gt_v_pred_xy = []
    APEs = []
    PEs = []  # percentage error (relative, so we retain the notion of over or under-prediction for the accuracy bias)
    PSs = []  # prediction stabilities as the ratio of within-mode predicted class over all predictions
    comb_pred = [[] for _ in range(len(CLASS_LIST))]

    # store gt and combined preds in additional csv file
    with open(os.path.join(OUTPUT_LOCATION, output_name + "---gt_vs_comb_pred.csv"), "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        if results_dict["inference_type"] == "REG":
            writer.writerow(["gt", "pred mean"])
        else:
            writer.writerow(["gt", "pred mode"])

        for key, value in ind_list.items():
            # originally the MEAN was used to get the "average" prediction
            # gt_v_pred_xy.append([value[0][0], np.mean([i[1] for i in value])])

            # use mode prediction when running classifier or detector, use mean prediction for regressor
            if results_dict["inference_type"] == "REG":
                pred_temp = np.mean([i[1] for i in value])
            else:
                pred_temp = sp.mode(np.array([i[1] for i in value]))[0]

            pred_std = np.std([i[1] for i in value])

            gt_temp = value[0][0]

            # write out combined prediction and gt
            writer.writerow([gt_temp, pred_temp])

            gt_v_pred_xy.append([gt_temp, pred_temp, pred_std])

            PE = 100 * (pred_temp - gt_temp) / gt_temp
            PEs.append(PE)

            APE = np.abs(PE)
            APEs.append(APE)

            comb_pred[find_class(CLASS_LIST, gt_temp)].append(APE)

            # get equivalent classes for regressor
            if results_dict["inference_type"] == "REG":
                pred_mode = find_class(CLASS_LIST, pred_temp)
                pred_all = [find_class(CLASS_LIST, i[1]) for i in value]
            else:
                pred_mode = pred_temp
                pred_all = [i[1] for i in value]

            PS = pred_all.count(pred_mode) / len(pred_all)
            PSs.append(PS)

    PS_out = np.nanmean(np.array(PSs))
    MAPE_true = np.nanmean(np.array(APEs))
    MAPE_std = np.nanstd(np.array(APEs))

    results_dict["MAPE_true"] = MAPE_true
    results_dict["MAPE_std"] = MAPE_std
    results_dict["Prediction_Stability"] = PS_out

    print("-----------------\n\n Prediction Stability:  ", PS_out)

    if create_plots:
        # next, create class-wise MAPE plots with the grouped predictions so re-compute class-wise MAPEs from extracted
        # mean / mode predictions above, so not accross all individual predictions but the cumulative prediction across
        # frames. This approach will stabilise spread in APEs and be consistent with reported MAPEs

        class_wise_MAPE = [np.mean(i) for i in comb_pred]
        class_wise_stdAPE = [np.std(i) for i in comb_pred]

        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['figure.dpi'] = 100
        fig, ax = plt.subplots()

        ax.errorbar(np.arange(len(CLASS_LIST)),
                    class_wise_MAPE,
                    class_wise_stdAPE, linestyle='None', marker='^', capsize=3)

        ax.set_ylabel('MAPE')
        ax.set_xticks(np.arange(len(CLASS_LIST)))
        ax.set_xticklabels(CLASS_LIST, rotation=45)
        ax.set_title('class-wise MAPE combined')
        ax.yaxis.grid(True)
        ax.set_yscale('log')
        ax.set_ylim(1, 5000)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---class-wise_MAPE_COMBINED.svg"), dpi='figure',
                    pad_inches=0.1)
        if verbose:
            plt.show()
        plt.close()

        # also, plot original style
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['figure.dpi'] = 100
        fig, ax = plt.subplots()

        ax.errorbar(np.arange(len(CLASS_LIST)),
                    class_wise_scores[:, 1],
                    class_wise_scores[:, 2], linestyle='None', marker='^', capsize=3)

        ax.set_ylabel('MAPE')
        ax.set_xticks(np.arange(len(CLASS_LIST)))
        ax.set_xticklabels(CLASS_LIST, rotation=45)
        ax.set_title('class-wise MAPE')
        ax.yaxis.grid(True)
        ax.set_yscale('log')
        ax.set_ylim(1, 5000)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---class-wise_MAPE.svg"), dpi='figure', pad_inches=0.1)
        if verbose:
            plt.show()
        plt.close()

    if create_plots:
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100
        fig, ax = plt.subplots()
        if known_ID:
            alpha = 0.1
        else:
            alpha = 0.4

    x = np.array([i[0] for i in gt_v_pred_xy])
    y = np.array([i[1] for i in gt_v_pred_xy])
    stds = np.array([i[2] for i in gt_v_pred_xy])

    try:
        results_dict["Spearman rank-order"] = sp.spearmanr(x, y).statistic
        results_dict["Spearman rank-order p-value"] = sp.spearmanr(x, y).pvalue

        results_dict["Precision bias"] = sp.spearmanr(x, coeff_var_ind).statistic
        results_dict["Precision bias p-value"] = sp.spearmanr(x, coeff_var_ind).pvalue

        results_dict["Absolute Accuracy bias"] = sp.spearmanr(x, np.array(APEs)).statistic
        results_dict["Absolute Accuracy bias p-value"] = sp.spearmanr(x, np.array(APEs)).pvalue

        results_dict["Relative Accuracy bias"] = sp.spearmanr(x, np.array(PEs)).statistic
        results_dict["Relative Accuracy bias p-value"] = sp.spearmanr(x, np.array(PEs)).pvalue
    except:
        print("=============\n\n SPEARMAN KILLED THE RADIO-STAR \n\n =============")

    # order_points = x.argsort()

    # x_ordered = x.copy()[order_points]
    # y_ordered = y.copy()[order_points]

    plot_limits = [0.0005, 0.06]

    if create_plots:
        ax.scatter(x, y, alpha=alpha)
        """
        ax.errorbar(x,
                    y,
                    stds, linestyle='None', marker='^', capsize=3, alpha=alpha)
        """
        ax.plot(plot_limits, plot_limits, '-k', linewidth=1.0)

    # parity plot statistics
    # Calculate Statistics of the Parity Plot
    # switched from linear fit to log scaled fit and reporting R^2 in log space

    log_x = np.log10(x)
    log_y = np.log10(y)

    mean_abs_err = np.mean(np.abs(x - y))
    rmse = np.sqrt(np.mean((x - y) ** 2))
    rmse_std = rmse / max(0.00001, np.std(y))

    z = np.polyfit(log_x, log_y, 1)

    log_x_extended = np.copy(log_x)
    log_x_extended = log_x_extended.tolist()
    log_x_extended.append(np.log10(plot_limits[1]))
    log_x_extended = np.asarray(log_x_extended)

    try:
        y_hat = np.power(10, np.poly1d(z)(log_x_extended))
    except ValueError:
        z = z.flatten()
        y_hat = np.power(10, np.poly1d(z)(log_x_extended))

    print("\n\n-----------------\n\n", z, "\n\n-----------------\n\n")

    msg = "Fitting line to gt v pred..."
    print(msg)
    output_txt.write(msg + "\n")

    slope = np.round(z[0], 4)
    msg = "slope       : " + str(slope)
    print(msg)
    output_txt.write(msg + "\n")

    y_intercept = np.round(z[1], 4)
    msg = "y intercept : " + str(y_intercept)
    print(msg)
    output_txt.write(msg + "\n")

    R_squared_log = round(metrics.r2_score(log_x, log_y), 4)
    R_squared_lin = round(metrics.r2_score(x, y), 4)

    msg = "R^2         : " + str(R_squared_log)
    print(msg)
    output_txt.write(msg + "\n")
    results_dict["R^2 log"] = R_squared_log
    results_dict["R^2 linear"] = R_squared_lin

    if create_plots:
        # create extended fitted line to match plot dimensions
        x_extended = np.copy(x)
        x_extended = x_extended.tolist()
        x_extended.append(plot_limits[1])
        x_extended = np.asarray(x_extended)

        ax.plot(x_extended, y_hat)

        text = f"$\: \: Mean \: Absolute \: Error \: (MAE) " \
               f"= {mean_abs_err:0.3f}$ \n $ Root \: Mean \: Square \: Error \: (RMSE) " \
               f"= {rmse:0.3f}$ \n $ RMSE \: / \: Std(y) = {rmse_std :0.3f}$ \n " \
               f"$R^2 = {metrics.r2_score(log_x, log_y):0.3f}$"

        plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')

        ax.set_ylabel('predicted weight [g]')
        ax.set_xlabel('ground truth weight [g]')
        ax.set_title('gt vs predicted weight')
        ax.yaxis.grid(True)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(0.0005, 0.06)
        ax.set_xlim(0.0005, 0.06)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_LOCATION, output_name + "---gt_vs_predicted_weight.svg"),
                    dpi='figure', pad_inches=0.1)

        if verbose:
            plt.show()
        plt.close()

    output_txt.write("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
    output_txt.close()

    return results_dict


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=str)
    ap.add_argument("-o", "--output", required=False, default="", type=str)
    ap.add_argument("-v", "--verbose", required=False, default=False, type=bool)
    ap.add_argument("-m", "--known_ID", required=False, default=False, type=bool)
    ap.add_argument("-gt", "--gt_from_name", required=False, default=False, type=bool)
    ap.add_argument("-d", "--detection_format", required=False, default=False, type=bool)

    args = vars(ap.parse_args())

    compute_scores(input_file=args["input"],
                   output=args["output"],
                   verbose=args["verbose"],
                   known_ID=args["known_ID"],
                   gt_from_name=args["gt_from_name"],
                   detection_format=args["detection_format"])
