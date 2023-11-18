import cv2
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import pandas as pd

sys.path.append("..")  # Adds higher directory to python modules path.

# now for the file management functions
from evaluation.Antrax_base import import_tracks, display_video, compose_video_with_overlay


def find_class(array, value):
    array_np = np.asarray(array)
    idx = (np.abs(array_np - value)).argmin()
    nearest_class = array_np[idx]
    pred_class = array.index(nearest_class)
    return pred_class


def clean_array(array, strip_NaN=True, strip_zero=True):
    array_np = np.asarray(array)
    if strip_NaN:
        array_np = array_np[np.logical_not(np.isnan(array_np))]
    if strip_zero:
        array_np = array_np[np.nonzero(array_np)]
    return array_np


input_video = "I:/EAEAAO/FOOTAGE/2019-08-03/2019-08-03_bramble-left.avi"
input_tracks = "I:/EAEAAO/Tracking/OUTPUT_TRACKS/2019-08-03_bramble-left"
input_poses = "I:/EAEAAO/POSES/2019-08-03_bramble-left"
input_file_order = "I:/EAEAAO/jobscripts/EAEAAO_batch_pose.pbs.o8454338.16"

start_frame = 0
end_frame = 1000

# read in order file and re-order tracks, so they correspond to their corresponding poses
file_order = open(input_file_order, 'r')
order_raw = file_order.readlines()

# Strips the newline character
order = []
for line in order_raw:
    if line.split("  with ")[0][-4:] == ".csv":
        order.append(line.split(" ")[2])
        print(order[-1])

# now we can load the captured video file and display it
cap = cv2.VideoCapture(input_video)

# check the number of frames of the imported video file
numFramesMax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("The imported clip:", input_video, "has a total of", numFramesMax, "frames.\n")

tracks = import_tracks(input_tracks, numFramesMax,
                       export=False,
                       min_track_length=100,
                       strip_tail_frames=50,
                       min_movement_px=50,
                       custom_order=order)

# The following function is used to display the tracks you imported.
# You can press "q" while hovering over the displayed video to exit.
print("\nDisplaying tracks loaded from:", input_tracks)
tracked_frames = len(tracks[0])

# next load all poses and append them to a list
all_poses = []
all_sizes = []
all_pose_ids = []

px_to_mm = 16.0848

for r, d, f in os.walk(input_poses):
    for file in f:
        print("Loading pose from", file)
        df = pd.read_csv(os.path.join(input_poses, file), delimiter=',', header=[0, 1, 2])

        x_diff = df["OmniTrax"]["b_t"]["x"].to_numpy() - df["OmniTrax"]["b_a_5"]["x"].to_numpy()
        y_diff = df["OmniTrax"]["b_t"]["y"].to_numpy() - df["OmniTrax"]["b_a_5"]["y"].to_numpy()

        lengths = np.sqrt(np.square(x_diff) + np.square(y_diff))
        median_length = np.round(np.median(lengths) / px_to_mm, 2)

        # now, use the appropriate class ID
        all_sizes.append(find_class([3, 4, 5, 6, 7], median_length))
        all_poses.append(df.to_numpy())
        all_pose_ids.append(int(file.split("_")[-1][:-4]))

all_poses_sorted = [pose for _, pose in sorted(zip(all_pose_ids, all_poses))]
all_sizes_sorted = [size for _, size in sorted(zip(all_pose_ids, all_sizes))]

compose_video_with_overlay(cap, tracks, poses=all_poses_sorted, scale=1.0, show=(16000, 18000),
                           size_classes=all_sizes_sorted, constant_frame_rate=False, pose_point_size=2,
                           video_name="LIGHT-MODE_" + input_video.split("/")[-1], DEBUG=True, thresh=0.5,
                           lightmode=True)
