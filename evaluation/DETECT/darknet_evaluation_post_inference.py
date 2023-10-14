import numpy as np
import pickle
import time
import sys
import os
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


def compare_frame(frame_gt, frame_detections, max_dist=0.05, network_shape=[None, None], confidence=0):
    """
    # NOTE: in this case we ignore the classification accuracy when assessing the mAP score!
    # therefore we strip class info from the detections and ground truth to assess the detector as a localizer
    # independently of its classification accuracy!
    """
    frame_detections = frame_detections[1]
    frame_gt = [i[1:] for i in frame_gt]

    # strip away all sub threshold detections!
    frame_detections = [f for f in frame_detections if f[1] > confidence]

    matches_gt = np.ones(len(frame_gt))
    matches_det = np.ones(len(frame_detections))
    below_thresh = 0
    detection_distances = []

    # now strip all empty entries from the ground truth

    for i in range(len(matches_gt)):
        min_dist = max_dist
        for j in range(len(matches_det)):

            if network_shape[0] is not None:
                norm_frame_detection = [frame_detections[j][2][0] / network_shape[0],
                                        frame_detections[j][2][1] / network_shape[1]]

            else:
                norm_frame_detection = frame_detections[j][2][0:2]

            match, px_dist = compare_points(gt=frame_gt[i][0:2],
                                            detection=norm_frame_detection,
                                            max_dist=max_dist)

            if match:
                matches_gt[i] = 0
                matches_det[j] = 0
                if px_dist < min_dist:
                    min_dist = px_dist

        if min_dist < max_dist:
            detection_distances.append(min_dist)

    missed_detections = int(np.sum(matches_gt))
    false_positives = int(np.sum(matches_det)) - below_thresh

    if len(detection_distances) == 0:
        mean_detection_distance = 0
    else:
        mean_detection_distance = np.mean(np.array(detection_distances))

    return len(frame_gt), missed_detections, false_positives, mean_detection_distance


def getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return int(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return int(os.popen('grep -c cores /proc/cpuinfo').read())


def process_detections(data):
    print("Running evaluation of ", data, "...")

    snapshots = [join(data, f) for f in listdir(data)]
    all_detections = []

    with open("ALL_ANNOTATIONS.pkl", 'rb') as f:
        all_annotations = pickle.load(f)

    for snapshot in snapshots:
        with open(snapshot, 'rb') as f:
            all_detections.append([snapshot, pickle.load(f)])

    print("ran inference on {} frames, using {}".format(len(all_detections[-1][1]), data))

    max_detection_distance_px = 0.1  # 0.1 = 10% away from centre to be considered a valid detection
    # thresh_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] (used in replicAnt paper)
    thresh_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    print("Computing AP scores for thresholds of {}".format(thresh_list))

    Results_mat = []

    # matrix shape: dataset(samples) , model x iteration x threshold

    print(all_detections[0][0])
    print("NUM DETECTIONS:", len(all_detections[0][1]))

    for model in all_detections:
        print("\n", model[0])

        Results_mat.append([model[0]])

        for confidence in thresh_list:
            num_frames_processed = 0
            Results_mat[-1].append([confidence])

            print("\n running inference at {} confidence threshold".format(confidence))

            for u, unique_dataset in enumerate(all_annotations):

                print("dataset:", unique_dataset[0])

                total_gt_detections = 0  # number of total detections in the ground truth dataset
                total_missed_detections = 0  # number of missed detections which are present in the ground truth dataset
                total_false_positives = 0  # number of incorrect detections that do not match any ground thruth tracks
                all_frame_detection_deviations = []  # list of mean deviations for correct detections

                for detection, annotation_local in zip(model[1][num_frames_processed:], unique_dataset[1:]):
                    gt_detections, missed_detections, false_positives, mean_detection_distance = compare_frame(
                        annotation_local,
                        detection,
                        max_detection_distance_px,
                        [800, 800],
                        confidence)

                    total_gt_detections += gt_detections
                    total_missed_detections += missed_detections
                    total_false_positives += false_positives
                    all_frame_detection_deviations.append(mean_detection_distance)

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


if __name__ == '__main__':
    # Data input and output
    modelFolder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\OUTPUT"
    outputFolder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\RESULTS"
    dataset = "I:\\WOLO\\BENCHMARK\\MultiCamAnts_YOLO\\data\\obj_test"
    DEBUG = False

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
