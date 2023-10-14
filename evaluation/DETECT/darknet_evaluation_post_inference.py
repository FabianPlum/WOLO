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


def compare_frame(frame_gt, frame_detections, max_dist=0.05, network_shape=[None, None], confidence=0, classes=20):
    """
    # NOTE: in this case we ignore the classification accuracy when assessing the mAP score!
    # therefore we strip class info from the detections and ground truth to assess the detector as a localizer
    # independently of its classification accuracy!
    """
    # LOOKUP tables for base CLASS and GT values

    base_class = ["0.0010", "0.0012", "0.0015", "0.0019", "0.0023", "0.0028", "0.0034",
                  "0.0042", "0.0052", "0.0064", "0.0078", "0.0096", "0.0118", "0.0145",
                  "0.0179", "0.0219", "0.0270", "0.0331", "0.0407", "0.0500"]

    # TODO -> add correct support for 5-class evaluation!
    # no worries, this can be easily done. The GT has all the uncompressed info!
    # pull the distinction between 5 and 20 class simply from the model name.
    base_class_five = ["0.0013", "0.0030", "0.0068", "0.0154", "0.0351"]
    lookup_class_five = [0, 0, 0, 0,
                         1, 1, 1, 1,
                         2, 2, 2, 2,
                         3, 3, 3, 3,
                         4, 4, 4, 4]

    gt_vals = {"plain": ["0.0010", "0.0012", "0.0016", "0.0020", "0.0025",
                         "0.0028", "0.0037", "0.0045", "0.0052", "0.0066",
                         "0.0077", "0.0094", "0.0122", "0.0143", "0.0193",
                         "0.0216", "0.0278", "0.0361", "0.0437", "0.0507"],
               "plain (fragments)": ["0.0010", "0.0013", "0.0015", "0.0020", "0.0023",
                                     "0.0028", "0.0037", "0.0040", "0.0051", "0.0063",
                                     "0.0075", "0.0092", "0.0118", "0.0147", "0.0175",
                                     "0.0240", "0.0293", "0.0358", "0.0431", "0.0495"],
               "brown forest floor": ["0.0006", "0.0012", "0.0014", "0.0019", "0.0023",
                                      "0.0030", "0.0033", "0.0041", "0.0051", "0.0069",
                                      "0.0078", "0.0093", "0.0123", "0.0137", "0.0192",
                                      "0.0213", "0.0264", "0.0363", "0.0403", "0.0505"],
               "dry soil": ["0.0004", "0.0013", "0.0017", "0.0020", "0.0023",
                            "0.0026", "0.0037", "0.0046", "0.0052", "0.0061",
                            "0.0081", "0.0098", "0.0129", "0.0155", "0.0183",
                            "0.0213", "0.0293", "0.0324", "0.0432", "0.0472"],
               "dry leaves": ["0.0010", "0.0012", "0.0014", "0.0019", "0.0023",
                              "0.0029", "0.0032", "0.0043", "0.0053", "0.0060",
                              "0.0085", "0.0092", "0.0115", "0.0156", "0.0177",
                              "0.0210", "0.0289", "0.0352", "0.0435", "0.0496"]}

    name_to_set = {'BROWN_FOREST_C920_SYNCHRONISED': "brown forest floor",
                   'BROWN_FOREST_DSLR_SYNCHRONISED': "brown forest floor",
                   'BROWN_FOREST_FLOOR_2023-04-10_16-05-32-05S': "brown forest floor",
                   'DRY_LEAVES_BACKGROUND_2023-04-11_14-13-59-13S': "dry leaves",
                   'DRY_LEAVES_C920_SYNCHRONISED': "dry leaves",
                   'DRY_LEAVES_DSLR_SYNCHRONISED': "dry leaves",
                   'DRY_SOIL_2023-04-12_12-20-03-20S': "dry soil",
                   'DRY_SOIL_C920_SYNCHRONISED': "dry soil",
                   'DRY_SOIL_DSLR_SYNCHRONISED': "dry soil",
                   'PLAIN_2023-04-10_13-29-27-29S': "plain",
                   'PLAIN_C920_SYNCHRONISED': "plain",
                   'PLAIN_DSLR_SYNCHRONISED': "plain",
                   'PLAIN_FRAGMENTS_C920_SYNCHRONISED': "plain (fragments)",
                   'PLAIN_FRAGMENTS_DSLR_SYNCHRONISED': "plain (fragments)",
                   'PLAIN_LEAF_FRAGMENTS_2023-04-11_12-12-23-12S': "plain (fragments)"
                   }

    file_base_name = frame_detections[0].split("\\")[-1][:-16]
    frame_num = frame_detections[0].split("frame_")[1][:-4]

    # strip away all sub threshold detections!
    frame_detections = [f for f in frame_detections[1] if f[1] > confidence]

    matches_gt = np.ones(len(frame_gt))
    matches_det = np.ones(len(frame_detections))
    below_thresh = 0
    detection_distances = []

    # now strip all empty entries from the ground truth
    highest_conf = 0

    class_output = [["", int(i[0]), -1, -1, -1] for i in frame_gt]

    for i in range(len(matches_gt)):
        # produce one row per ground truth to produce output table
        # class predictions should have the following contents to allow for reusing the same inference style as for the
        # regression and classifications stream:
        # class_output = [file (test/gt_glass/video-file_gt.jpg), gt_class, pred_class, gt, pred]
        # generating this structure means the array can be sorted by filenames to create an identical data structure
        gt_class = int(frame_gt[i][0])
        base_class_temp = base_class[gt_class]
        gt = gt_vals[name_to_set[file_base_name]][gt_class]

        if classes == 5:
            # overwrite gt classes for 5 class case
            gt_class = lookup_class_five[int(frame_gt[i][0])]

        file_name_sample = "test/" + base_class_temp.replace(".", "") + "/" + file_base_name + "_" + \
                           frame_num + "_" + gt.replace(".", "") + ".jpg"

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

            match, px_dist = compare_points(gt=frame_gt[i][1:3],
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

        class_output[i] = [file_name_sample, gt_class, pred[0], gt, pred[1]]

    missed_detections = int(np.sum(matches_gt))
    false_positives = int(np.sum(matches_det)) - below_thresh

    if len(detection_distances) == 0:
        mean_detection_distance = 0
    else:
        mean_detection_distance = np.mean(np.array(detection_distances))

    return len(frame_gt), missed_detections, false_positives, mean_detection_distance, class_output


def getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return int(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return int(os.popen('grep -c cores /proc/cpuinfo').read())


def process_detections(data):
    REGENERATE_AP_SCORES = False
    print("Running evaluation of ", data, "...")
    num_classes = int(data.split("_")[-1])
    print("INFO: num_classes:", num_classes)

    if not REGENERATE_AP_SCORES:
        print("WARNING: GENERATING AP SCORES IS DISABLED!")
        thresh_list = [0.5]  # (used in WOLO for classification comparison)
    else:
        # (used in replicAnt paper)
        # thresh_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        # (used in WOLO for AP only)
        thresh_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    snapshots = [join(data, f) for f in listdir(data)]
    all_detections = []

    with open("ALL_ANNOTATIONS.pkl", 'rb') as f:
        all_annotations = pickle.load(f)

    for snapshot in snapshots:
        with open(snapshot, 'rb') as f:
            all_detections.append([snapshot, pickle.load(f)])

    print("ran inference on {} frames, using {}".format(len(all_detections[-1][1]), data))

    max_detection_distance_px = 0.05  # 0.1 = 10% away from centre to be considered a valid detection

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

            for u, unique_dataset in enumerate(all_annotations):

                print("dataset:", unique_dataset[0], "using model", data, "and threshold", confidence)

                total_gt_detections = 0  # number of total detections in the ground truth dataset
                total_missed_detections = 0  # number of missed detections which are present in the ground truth dataset
                total_false_positives = 0  # number of incorrect detections that do not match any ground thruth tracks
                all_frame_detection_deviations = []  # list of mean deviations for correct detections

                for detection, annotation_local in zip(model[1][num_frames_processed:], unique_dataset[1:]):
                    gt_detections, missed_detections, false_positives, mean_detection_distance, co = compare_frame(
                        annotation_local,
                        detection,
                        max_detection_distance_px,
                        [800, 800],
                        confidence,
                        classes=num_classes)

                    total_gt_detections += gt_detections
                    total_missed_detections += missed_detections
                    total_false_positives += false_positives
                    all_frame_detection_deviations.append(mean_detection_distance)

                    class_outputs += co

                    num_frames_processed += 1

                mean_px_error = np.mean(all_frame_detection_deviations) * 100
                detection_accuracy = ((
                                              total_gt_detections - total_missed_detections - total_false_positives) / total_gt_detections) * 100

                if total_gt_detections == total_missed_detections:
                    # the accuracy is zero if no objects are correctly detected
                    AP = 0
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

                Results_mat[-1][-1].append([unique_dataset[0],
                                            total_gt_detections,
                                            total_gt_detections - total_missed_detections,
                                            total_missed_detections,
                                            total_false_positives,
                                            AP,
                                            Recall])

    outputFolder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\RESULTS"
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
    # Data input and output
    modelFolder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\OUTPUT"
    outputFolder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\RESULTS"
    dataset = "I:\\WOLO\\BENCHMARK\\MultiCamAnts_YOLO\\data\\obj_test"
    DEBUG = False
    REGENERATE_ANNOTATIONS = True

    if REGENERATE_ANNOTATIONS:
        # structure of ground truth data
        all_annotations = []
        unique_datasets = []

        # get test sample files
        sample_paths = [join(dataset, f) for f in listdir(dataset)
                        if str(join(dataset, f)).split(".")[-1] == "jpg"]

        print("\nFound a total of %s samples" % len(sample_paths))

        print("\nFolder contains the following sub-datasets:\n")
        for sample in sample_paths:
            annotation = sample.split(".")[0] + ".txt"
            dataset_name = "_".join(str(os.path.basename(sample)).split("_")[:-1])

            # get all unique datasets from input folder
            if dataset_name not in unique_datasets:
                unique_datasets.append(dataset_name)
                all_annotations.append([dataset_name])
                print(all_annotations[-1])

            f = open(annotation, 'r')
            Lines = f.readlines()

            bounding_boxes = []  # c,x,y,w,h
            # Strips the "\n" newline character
            for line in Lines:
                line_cleaned = line.strip()
                line_arr = line_cleaned.split(" ")
                bounding_boxes.append([float(f) for f in line_arr])

            all_annotations[-1].append(bounding_boxes)
            f.close()

        print("\nLoaded %s annotated sub-datasets in total" % len(all_annotations))

        with open("ALL_ANNOTATIONS.pkl", 'wb') as f:
            pickle.dump(all_annotations, f)

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

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
