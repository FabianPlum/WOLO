import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os
import sys
import cv2

sys.path.append("..")  # Adds higher directory to python modules path.

# now for the file management functions
from evaluation.Antrax_base import import_tracks


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


def get_derivative(tracks, time_step=1 / 30):
    tracks_der = np.zeros([tracks.shape[0], int((tracks.shape[1] - 1) / 2)])
    # convert tracks to float array
    tracks_der = tracks_der.astype(float)

    num_tracks = int((tracks.shape[1] - 1) / 2)
    num_timeSteps = tracks.shape[0]

    print("Computing derivative for", num_tracks, "tracks...")

    for t in range(num_timeSteps - 1):
        # then retrieve all track centres at the given frame
        track_centres = tracks[t, 1:]
        track_centres_non_zero_ids = np.nonzero(track_centres)[0][::2]

        if len(track_centres_non_zero_ids) != 0:

            for track_orig in track_centres_non_zero_ids:
                track = int((track_orig + 1) / 2)

                # check that the next value is no zero

                if tracks[t + 1, track * 2 + 1] == 0 or tracks[t + 1, track * 2 + 2] == 0:
                    continue

                tracks_der[t, track] = np.sqrt((tracks[t + 1, track * 2 + 1] -
                                                tracks[t, track * 2 + 1]) ** 2 +
                                               (tracks[t + 1, track * 2 + 2] -
                                                tracks[t, track * 2 + 2]) ** 2) / time_step

    return tracks_der


def apply_moving_average(tracks, samples=3, padding_mode='valid'):
    tracks_ma = np.copy(tracks)
    tracks_ma.astype(float)
    num_tracks = int((tracks.shape[1] - 1) / 2)

    print("Applying moving average of size", samples, "to", num_tracks, "tracks...")
    for t in range(num_tracks):
        # apply smoothing only to areas where tracked subject is in view
        # therefore, all zero entries must be ignored
        track_temp_x = tracks[:, t * 2 + 1]
        track_temp_y = tracks[:, t * 2 + 2]

        nonzeroind = np.nonzero(track_temp_x)[0]

        # the following code is bad and slow. But I don't care. There is a strange artifact in some exported tracks
        # where there are sudden un-connected coordinates appearing outside the tracked area. So we find those and
        # remove them. No matter the (compute) cost.

        rm_ind = []

        for ind, f in enumerate(nonzeroind):
            if 0 < f < len(tracks) - 1:
                if tracks[f, t * 2 + 1] != 0 and tracks[f - 1, t * 2 + 1] == 0 and tracks[f + 1, t * 2 + 1] == 0:
                    rm_ind.append(ind)

        nonzeroind = np.delete(nonzeroind, rm_ind)

        track_x = np.convolve(track_temp_x[nonzeroind[0]:nonzeroind[-1] - samples], np.ones(samples) / samples,
                              mode=padding_mode)
        track_y = np.convolve(track_temp_y[nonzeroind[0]:nonzeroind[-1] - samples], np.ones(samples) / samples,
                              mode=padding_mode)

        tracks_ma[nonzeroind[0] + math.floor(samples / 2):nonzeroind[-1] - math.ceil(samples / 2) - samples + 1,
        t * 2 + 1] = track_x
        tracks_ma[nonzeroind[0] + math.floor(samples / 2):nonzeroind[-1] - math.ceil(samples / 2) - samples + 1,
        t * 2 + 2] = track_y

        # add back zeros
    return tracks_ma


# as we are going to plot all speeds across all animals in all videos, we can use a singular list of all relevant
# videos, assuming they all have roughly the same magnification and frame of view (which is reasonable here)

video_list = ["2019-07-22/2019-07-22_bramble_left.mp4",
              "2019-07-23/2019-07-23_bramble_right2.avi",
              "2019-07-23/2019-07-23_bramble_right.avi",
              "2019-07-23/2019-07-23_rose_left_2.avi",
              "2019-07-23/2019-07-23_rose_left.avi",
              "2019-07-24/2019-07-24_bramble_left.avi",
              "2019-07-24/2019-07-24_bramble_right.avi",
              "2019-07-25/2019-07-25_rose_left.avi",
              "2019-07-25/2019-07-25_rose_right.avi",
              "2019-07-30/2019-07-30_bramble_left.avi",
              "2019-07-30/2019-07-30_rose_right.avi",
              "2019-07-31/2019-07-31_bramble_left.avi",
              "2019-07-31/2019-07-31_bramble_right.avi",
              "2019-08-01/2019-08-01_bramble_left.avi",
              "2019-08-01/2019-08-01_rose_right.avi",
              "2019-08-03/2019-08-03_bramble-left.avi",
              "2019-08-03/2019-08-03_bramble-right.avi",
              "2019-08-05/2019-08-05_bramble_left.avi",
              "2019-08-05/2019-08-05_rose_right.avi",
              "2019-08-06/2019-08-06_bramble_left.avi",
              "2019-08-06/2019-08-06_rose_right.avi",
              "2019-08-07/2019-08-07_bramble_left.avi",
              "2019-08-07/2019-08-07_bramble_right.avi",
              "2019-08-08/2019-08-08_rose_left.avi",
              "2019-08-08/2019-08-08_rose_right.avi",
              "2019-08-09/2019-08-09_bramble_left.avi",
              "2019-08-09/2019-08-09_rose_right.avi",
              "2019-08-12/2019-08-12_rose_left.avi",
              "2019-08-12/2019-08-12_rose_right.avi",
              "2019-08-13/2019-08-13_bramble_right.avi",
              "2019-08-13/2019-08-13_rose_left.avi",
              "2019-08-15/2019-08-15_bramble_left.avi",
              "2019-08-15/2019-08-15_bramble_right.avi",
              "2019-08-16/2019-08-16_bramble_right.avi",
              "2019-08-16/2019-08-16_rose_left.avi",
              "2019-08-20/2019-08-20_rose_left.avi",
              "2019-08-20/2019-08-20_rose_right.avi",
              "2019-08-21/2019-08-21_rose_left.avi",
              "2019-08-21/2019-08-21_rose_right.avi",
              "2019-08-22/2019-08-22_bramble_right.avi",
              "2019-08-22/2019-08-22_rose_left.avi"]

video_absolute_path = "I:/EAEAAO/FOOTAGE"
tracks_absolute_path = "J:/OUTPUT_TRACKS"
weights_absolute_path = "I:/EAEAAO/RESULTS/weight_estimates_CLASS_MultiCamAnts-and-synth-simple_5_sigma-2_cross-entropy"

weight_class_v_speed = []  # each entry here has the shape [pred_class, max_speed] -> shape = [n,2]

# collect tracks and weights, working through one set at a time, extending weight_class_v_speed

for input_file in video_list:
    video_path = os.path.join(video_absolute_path, input_file)
    tracks_path = os.path.join(tracks_absolute_path, input_file.split("/")[-1][:-4])
    weights_path = os.path.join(weights_absolute_path, input_file.split("/")[-1][:-4] + "_ALL_WEIGHTS.pickle")

    # first import all corresponding estimated weights

    with open(weights_path, 'rb') as pickle_file:
        all_weights_temp = pickle.load(pickle_file)
        all_weights_compressed_median = np.round(np.nanmedian(all_weights_temp, axis=0), 4)

    five_class = [0.0013, 0.0030, 0.0068, 0.0154, 0.0351]
    five_class_limits = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

    # produced using median
    pred_classes = [find_class(five_class, float(x)) for x in all_weights_compressed_median]
    pred_classes = clean_array(pred_classes, strip_NaN=True)

    print("\n", input_file, "contains", pred_classes.shape[0], "estimated weights.")

    out_array = np.zeros((pred_classes.shape[0], 2))
    out_array[:, 0] = pred_classes

    # now retrieve all corresponding tracks and compute their max speed

    min_track_length = 100  # set to 0 if all tracks should be used, regardless of length
    strip_tail_frames = 50  # set to zero if all are kept (used to strip extrapolation)
    min_movement_px = 50  # controls how "still" a subject may be before its excluded

    # now we can load the captured video file and display it
    cap = cv2.VideoCapture(video_path)

    # check the number of frames of the imported video file
    numFramesMax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("The imported clip:", video_path, "has a total of", numFramesMax, "frames.\n")

    tracks = (import_tracks(tracks_path, numFramesMax,
                            export=False,
                            min_track_length=min_track_length,
                            strip_tail_frames=strip_tail_frames,
                            min_movement_px=min_movement_px,
                            VERBOSE=False))

    print(input_file, "contains", int((tracks.shape[1] - 1) / 2), "valid tracks")

    tracks_MA = apply_moving_average(tracks, 5)

    tracks_speed = get_derivative(tracks_MA, time_step=1 / 25)

    mean_speed = np.mean(tracks_speed[np.nonzero(tracks_speed)], axis=0)

    out_array[:, 1] = mean_speed

    weight_class_v_speed.extend(out_array.tolist())

    # export all_tracks and all_weights
    with open(str(os.path.basename(input_file))[:-4] + "_WEIGHTS_AND_MEAN_SPEED_IN_PX_per_S.pickle", 'wb') as handle:
        pickle.dump(out_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

weight_class_v_speed_np = np.array(weight_class_v_speed)

print("\nTotal estimates n = ", weight_class_v_speed_np.shape[0])

with open("ALL_WEIGHTS_AND_SPEED_MEAN_IN_PX_per_S.pickle", 'wb') as handle:
    pickle.dump(weight_class_v_speed_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
