import numpy as np
import pickle
import time
import sys
import os
import pandas as pd
from os import listdir
from os.path import join
from scipy.spatial import distance
from multiprocessing import Pool


def compare_points(gt, detection, max_dist=25):
    match = False
    px_distance = distance.euclidean(gt, detection)
    if px_distance <= max_dist:
        match = True
    return match, px_distance


def compare_frame(frame_detections, max_dist=0.05, network_shape=[None, None], confidence=0, classes=20):
    """
    # NOTE: in this case we ignore the classification accuracy when assessing the mAP score!
    # therefore we strip class info from the detections and ground truth to assess the detector as a localizer
    # independently of its classification accuracy!
    """
    # LOOKUP tables for base CLASS and GT values

    base_class = ["0.0010", "0.0012", "0.0015", "0.0019", "0.0023", "0.0028", "0.0034",
                  "0.0042", "0.0052", "0.0064", "0.0078", "0.0096", "0.0118", "0.0145",
                  "0.0179", "0.0219", "0.0270", "0.0331", "0.0407", "0.0500"]

    base_class_five = ["0.0013", "0.0030", "0.0068", "0.0154", "0.0351"]
    lookup_class_five = [0, 0, 0, 0,
                         1, 1, 1, 1,
                         2, 2, 2, 2,
                         3, 3, 3, 3,
                         4, 4, 4, 4]

    full_sample_name = frame_detections[0].split("\\")[-3:]
    file_base_name = frame_detections[0].split("\\")[-1][:-4]
    class_name = frame_detections[0].split("\\")[-2]

    try:
        gt = float(file_base_name.split(" ")[-1])
    except ValueError:
        gt = float(file_base_name.split("-")[-1])

    # print(file_base_name)
    # print(gt)
    # print(class_name)

    if gt > 1:
        gt = int(gt / 10000)

    gt_class = base_class.index("0.0" + class_name[2:])
    # print(gt_class)

    if classes == 5:
        # overwrite gt classes for 5 class case
        gt_class = lookup_class_five[gt_class]

    # TODO -> get 20 v 5 class info, then get **gt_class** from lookup table
    # TODO -> get **pred** and **pred_class** from centre-most predictions (within 0.1 width) with highest activation
    # TODO -> return only one estimate (max) per image

    # strip away all sub threshold detections!
    frame_detections = [f for f in frame_detections[1] if f[1] > confidence]

    # print(frame_detections)

    matches_gt = [1]
    matches_det = np.ones(len(frame_detections))
    below_thresh = 0
    detection_distances = []

    # now strip all empty entries from the ground truth
    highest_conf = 0

    for i in range(len(matches_gt)):
        # produce one row per ground truth to produce output table
        # class predictions should have the following contents to allow for reusing the same inference style as for the
        # regression and classifications stream:
        # class_output = [file (test/gt_glass/video-file_gt.jpg), gt_class, pred_class, gt, pred]
        # generating this structure means the array can be sorted by filenames to create an identical data structure

        min_dist = max_dist
        # only keep the locally highest activated prediction, if multiple exist for one animal (nms)
        conf = highest_conf
        pred = [-1, -1]
        for j in range(len(matches_det)):

            if network_shape[0] is not None:
                norm_frame_detection = [frame_detections[j][2][0] / network_shape[0],
                                        frame_detections[j][2][1] / network_shape[1]]

            else:
                norm_frame_detection = frame_detections[j][2][0:2]

            # assume the individual to be located in the centre of the frame
            match, px_dist = compare_points(gt=[0.5, 0.5],
                                            detection=norm_frame_detection,
                                            max_dist=max_dist)

            if match:
                matches_gt[i] = 0
                matches_det[j] = 0
                if px_dist < min_dist:
                    min_dist = px_dist

                # record class of the highest confidence match
                if float(frame_detections[j][1]) > conf:
                    conf = float(frame_detections[j][1])
                    pred_temp = str(frame_detections[j][0])[2:-1]
                    while len(pred_temp) < 6:
                        # pad values to fit naming convention
                        pred_temp += "0"

                    if classes == 5:
                        pred[0] = base_class_five.index(pred_temp)
                    else:
                        pred[0] = base_class.index(pred_temp)
                    pred[1] = pred_temp

        if min_dist < max_dist:
            detection_distances.append(min_dist)

        class_output = ["/".join(full_sample_name), gt_class, pred[0], gt, pred[1]]

        # print(class_output)

    missed_detections = int(np.sum(matches_gt))
    false_positives = int(np.sum(matches_det)) - below_thresh

    if len(detection_distances) == 0:
        mean_detection_distance = 0
    else:
        mean_detection_distance = np.mean(np.array(detection_distances))

    return 1, missed_detections, false_positives, mean_detection_distance, class_output


def getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return int(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return int(os.popen('grep -c cores /proc/cpuinfo').read())


def process_detections(data):
    dataset_name = "DSLR_C920_CLASS_128"  # "CORVIN9000_128"

    print("Running evaluation of ", data, "...")
    num_classes = int(data.split("_")[-1])
    print("INFO: num_classes:", num_classes)

    thresh_list = [0.1]  # (used in WOLO for CROPPED classification comparison)

    snapshots = [join(data, f) for f in listdir(data)]
    all_detections = []

    for snapshot in snapshots:
        # only use the snapshot of the respective CROPPED TEST dataset
        if snapshot.split(dataset_name)[-1] == ".pkl":
            with open(snapshot, 'rb') as f:
                all_detections.append([snapshot, pickle.load(f)])

    print("ran inference on {} frames, using {}".format(len(all_detections[-1][1]), data))

    max_detection_distance_px = 0.2  # = 10% away from centre to be considered a valid detection

    print("Computing AP scores for thresholds of {}".format(thresh_list))

    Results_mat = []

    # matrix shape: dataset(samples) , model x iteration x threshold

    print(all_detections[0][0])
    print("NUM DETECTIONS:", len(all_detections[0][1]))

    class_outputs = []

    for model in all_detections:
        print("\n", model[0])

        Results_mat.append([model[0]])

        for confidence in thresh_list:
            num_frames_processed = 0
            Results_mat[-1].append([confidence])

            print("\n running inference at {} confidence threshold".format(confidence))

            print("dataset:", dataset_name, "using model", data, "and threshold", confidence)

            total_gt_detections = 0  # number of total detections in the ground truth dataset
            total_missed_detections = 0  # number of missed detections which are present in the ground truth dataset
            total_false_positives = 0  # number of incorrect detections that do not match any ground thruth tracks
            all_frame_detection_deviations = []  # list of mean deviations for correct detections

            for detection in model[1]:
                gt_detections, missed_detections, false_positives, mean_detection_distance, co = compare_frame(
                    detection,
                    max_detection_distance_px,
                    [128, 128],  # [64,64] # [800,800]
                    confidence,
                    classes=num_classes)

                total_gt_detections += gt_detections
                total_missed_detections += missed_detections
                total_false_positives += false_positives
                all_frame_detection_deviations.append(mean_detection_distance)

                if confidence == 0.1:
                    # only report class predictions at 50% confidence for comparison
                    class_outputs.append(co)

                num_frames_processed += 1

            mean_px_error = np.mean(all_frame_detection_deviations) * 100
            detection_accuracy = ((
                                          total_gt_detections - total_missed_detections - total_false_positives) / total_gt_detections) * 100

            if total_gt_detections == total_missed_detections:
                # the average precision and recall are zero if no objects are correctly detected
                AP = 0
                Recall = 0
            else:
                AP = (total_gt_detections - total_missed_detections) / (
                        total_gt_detections - total_missed_detections + total_false_positives)
                Recall = (total_gt_detections - total_missed_detections) / total_gt_detections

            print("Total ground truth detections:", total_gt_detections)
            print("Total correct detections:", total_gt_detections - total_missed_detections)
            print("Total missed detections:", total_missed_detections)
            print("Total false positives:", total_false_positives)
            print("Average Precision:", round(AP, 3))
            print("Recall:", round(Recall, 3))
            print("Detection accuracy (GT - FP - MD) / GT):", np.round(detection_accuracy, 1), "%")
            print("Mean relative deviation: {} %\n".format(np.round(mean_px_error, 3)))

            Results_mat[-1][-1].append([dataset_name,
                                        total_gt_detections,
                                        total_gt_detections - total_missed_detections,
                                        total_missed_detections,
                                        total_false_positives,
                                        AP,
                                        Recall])

    outputFolder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\RESULTS_TEST_128x128_DSLR_C920"
    output_results = join(outputFolder, str(os.path.basename(data)) + "_RESULTS.pkl")
    print(output_results)

    with open(output_results, 'wb') as fp:
        pickle.dump(Results_mat, fp)

    out_df = pd.DataFrame({"file": [i[0] for i in class_outputs],
                           "gt_class": [i[1] for i in class_outputs],
                           "pred_class": [i[2] for i in class_outputs],
                           "gt": [i[3] for i in class_outputs],
                           "pred": [i[4] for i in class_outputs]})

    # sort dataframe by file name so classes and datasets are grouped
    out_df_sorted = out_df.sort_values("file")

    out_df_sorted.to_csv(join(outputFolder, str(os.path.basename(data)) + "_test_data_pred_results.csv"))


if __name__ == '__main__':
    modelFolder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\OUTPUT"
    outputFolder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\RESULTS_TEST_128x128_DSLR_C920"
    DEBUG = False

    start_time = time.time()

    """
    
    AND NOW EVALUATE ALL OF THIS!
    
    """

    # get all model paths
    model_paths = [join(modelFolder, f) for f in listdir(modelFolder)]
    model_paths.sort()

    print("\nFound a total of {} trained models".format(len(model_paths)))

    # each model should be handled by a separate thread, regardless of the number of snapshots in there.
    # As soon as a thread is done with one model it should move on to the next.
    num_threads = getThreads()
    print("Running %s parallel threads to evaluate data..." % num_threads)

    if DEBUG:
        for model_name in model_paths:
            process_detections(model_name)

    else:
        with Pool(num_threads) as p:
            p.map(process_detections, model_paths)

    print("\n--- process completed within %s seconds ---" % round(time.time() - start_time, 2))
