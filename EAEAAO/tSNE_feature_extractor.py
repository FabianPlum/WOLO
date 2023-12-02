import argparse
import cv2
import pickle
import os
from tqdm import tqdm
import numpy as np
import sys
import pandas as pd
from umap import UMAP
from sklearn.manifold import TSNE
import plotly.express as px
import math

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


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def get_derivative(tracks, time_step=1 / 30, verbose=False):
    tracks_der = np.zeros([tracks.shape[0], int((tracks.shape[1] - 1) / 2)])
    # convert tracks to float array
    tracks_der = tracks_der.astype(float)

    num_tracks = int((tracks.shape[1] - 1) / 2)
    num_timeSteps = tracks.shape[0]

    if verbose:
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


def get_derivative_angles(tracks, time_step=1 / 30, verbose=False):
    tracks = tracks[:, 1:]  # strip frame counter
    tracks_der = np.zeros((tracks.shape[0] - 1, tracks.shape[1]))
    # convert tracks to float array
    tracks_der = tracks_der.astype(float)

    num_tracks = tracks.shape[1]
    num_timeSteps = tracks.shape[0]

    if verbose:
        print("Computing derivative for", num_tracks, "angles...")

    for t in range(num_timeSteps - 1):
        # then retrieve all track centres at the given frame

        for track in range(num_tracks):
            # check that the next value is non-zero

            if tracks[t + 1, track] == 0:
                continue

            tracks_der[t, track] = (tracks[t + 1, track] - tracks[t, track]) / time_step

    return tracks_der


def apply_moving_average(tracks, samples=3, padding_mode='valid', verbose=False):
    tracks_ma = np.copy(tracks)
    tracks_ma.astype(float)
    num_tracks = int((tracks.shape[1] - 1) / 2)

    if verbose:
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


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-t", "--tracks", required=True, type=str)
    ap.add_argument("-v", "--video", required=True, type=str)
    ap.add_argument("-p", "--poses", required=True, type=str)
    ap.add_argument("-o", "--order", required=True, type=str)

    # optional parameters
    ap.add_argument("-tr", "--threshold", default=0.5, required=False, type=float)
    ap.add_argument("-cl", "--clip_length", default=25, required=False, type=int)
    ap.add_argument("-d", "--display", default=False, required=False, type=bool)
    ap.add_argument("-out", "--output_folder", default="", required=False, type=str)
    ap.add_argument("-test", "--stop_early", default=False, required=False, type=bool)

    args = vars(ap.parse_args())

    input_video = args["video"]
    input_tracks = args["tracks"]
    input_poses = args["poses"]
    input_file_order = args["order"]

    clip_length = args["clip_length"]  # meaning, we consider each 30-frame instance as one behavioural sample
    stop_early = args["stop_early"]

    output_folder = args["output_folder"]

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

    px_to_mm_POSE = 16.0848
    px_to_mm_TRACK = 6.8628

    for r, d, f in os.walk(input_poses):
        for file in f:
            print("Loading pose from", file)
            df = pd.read_csv(os.path.join(input_poses, file), delimiter=',', header=[0, 1, 2])

            x_diff = df["OmniTrax"]["b_t"]["x"].to_numpy() - df["OmniTrax"]["b_a_5"]["x"].to_numpy()
            y_diff = df["OmniTrax"]["b_t"]["y"].to_numpy() - df["OmniTrax"]["b_a_5"]["y"].to_numpy()

            lengths = np.sqrt(np.square(x_diff) + np.square(y_diff))
            median_length = np.round(np.median(lengths) / px_to_mm_POSE, 2)

            # now, use the appropriate class ID
            all_sizes.append(find_class([3, 4, 5, 6, 7], median_length))
            all_poses.append(df.to_numpy())
            all_pose_ids.append(int(file.split("_")[-1][:-4]))

    all_poses_sorted = [pose for _, pose in sorted(zip(all_pose_ids, all_poses))]
    all_sizes_sorted = [size for _, size in sorted(zip(all_pose_ids, all_sizes))]

    # now loop over all pose elements and produce the format stated above
    # ["video", "material", "startframe", "id", "speed", "size_class"]

    material = input_poses.split("_")[-1].split("-")[0]
    out_list = []
    centre_poses = True  # if true, centres poses at b_t
    align_poses = True  # if true, aligns all poses with b_t -> b_a_1 being the middle
    display = False

    compute_size_class_for_each_action = True

    track_id = -1  # we can now iterate through tracks consecutively as we reordered them based on their associated pose

    for id, pose, size in tqdm(zip(order, all_poses_sorted, all_sizes_sorted), total=(len(order))):
        track_id += 1
        id_val = int(id.split("_")[-1][:-4])
        if stop_early:
            if track_id == 100:
                break
        for instances in range(math.floor(int(len(pose) / clip_length))):
            skip_instance = False

            num_features = pose[int(instances * clip_length) + 1:int(instances * clip_length) + 1 + clip_length, 1:]
            num_features = int(num_features.shape[1] / 3)

            # drop_thresh_idx = [i*3 + 2 for i in range(num_features)]

            frame_start = int(pose[int(instances * clip_length) + 1, 0])
            # now create feature vector from instance
            features = pose[int(instances * clip_length) + 1:int(instances * clip_length) + 1 + clip_length, 1:]

            out_features = np.zeros(num_features * 2 * clip_length)
            out_features_angles = np.zeros(9 * clip_length)  # abdomen, legs, antennae

            loc_counter = 0

            for frame in range(clip_length):
                if display:
                    blank_image = np.zeros((300, 300, 3), np.uint8)

                try:
                    avg_front = np.nanmean([features[frame][0:2], features[frame][126:128]], axis=0)
                    avg_back = np.nanmean(
                        [features[frame][3:5], features[frame][6:8], features[frame][9:11], features[frame][12:14]],
                        axis=0)
                    body_orientation = np.arccos(np.dot([0, 1], avg_back - avg_front) /
                                                 (np.linalg.norm([0, 1]) * np.linalg.norm(avg_back - avg_front)))
                except IndexError:
                    skip_instance = True
                    break

                for p in range(num_features):
                    if features[frame][p * 3 + 2] >= 0.5:
                        point = features[frame][p * 3:p * 3 + 2]

                        if centre_poses:
                            point = point - avg_front + [150, 120]

                        if align_poses:
                            point = rotate(point, [150, 120], degrees=body_orientation * 180 / np.pi)

                        # print(loc_counter)
                        out_features[loc_counter:loc_counter + 2] = point

                        if display:
                            blank_image = cv2.circle(blank_image, (int(point[0]), int(point[1])),
                                                     3,
                                                     (int(255 * p / num_features),
                                                      int(255 - 255 * p / num_features), 200),
                                                     -1)
                    loc_counter += 2

                # get joint angles and use the weighted mean of each leg for more robust estimates
                # the following is hard coded to get locations of key body parts but that's okay for now.
                # It works for insects.
                loc_antennae_l = np.reshape(features[frame][138:147], (-1, 3))
                loc_antennae_r = np.reshape(features[frame][129:138], (-1, 3))
                loc_leg_l_1 = np.reshape(features[frame][72:90], (-1, 3))
                loc_leg_l_2 = np.reshape(features[frame][90:108], (-1, 3))
                loc_leg_l_3 = np.reshape(features[frame][108:126], (-1, 3))
                loc_leg_r_1 = np.reshape(features[frame][18:36], (-1, 3))
                loc_leg_r_2 = np.reshape(features[frame][36:54], (-1, 3))
                loc_leg_r_3 = np.reshape(features[frame][54:72], (-1, 3))
                loc_abdomen = np.reshape(features[frame][3:18], (-1, 3))

                avg_loc_antennae_l = np.average(loc_antennae_l[:, :2], axis=0,
                                                weights=np.transpose([loc_antennae_l[:, 2], loc_antennae_l[:, 2]]))
                avg_loc_antennae_r = np.average(loc_antennae_r[:, :2], axis=0,
                                                weights=np.transpose([loc_antennae_r[:, 2], loc_antennae_r[:, 2]]))
                avg_loc_leg_l_1 = np.average(loc_leg_l_1[:, :2], axis=0,
                                             weights=np.transpose([loc_leg_l_1[:, 2], loc_leg_l_1[:, 2]]))
                avg_loc_leg_l_2 = np.average(loc_leg_l_2[:, :2], axis=0,
                                             weights=np.transpose([loc_leg_l_2[:, 2], loc_leg_l_2[:, 2]]))
                avg_loc_leg_l_3 = np.average(loc_leg_l_3[:, :2], axis=0,
                                             weights=np.transpose([loc_leg_l_3[:, 2], loc_leg_l_3[:, 2]]))
                avg_loc_leg_r_1 = np.average(loc_leg_r_1[:, :2], axis=0,
                                             weights=np.transpose([loc_leg_r_1[:, 2], loc_leg_r_1[:, 2]]))
                avg_loc_leg_r_2 = np.average(loc_leg_r_2[:, :2], axis=0,
                                             weights=np.transpose([loc_leg_r_2[:, 2], loc_leg_r_2[:, 2]]))
                avg_loc_leg_r_3 = np.average(loc_leg_r_3[:, :2], axis=0,
                                             weights=np.transpose([loc_leg_r_3[:, 2], loc_leg_r_3[:, 2]]))
                avg_loc_abdomen = np.average(loc_abdomen[:, :2], axis=0,
                                             weights=np.transpose([loc_abdomen[:, 2], loc_abdomen[:, 2]]))

                combined_poses = [[loc_antennae_l, avg_loc_antennae_l],
                                  [loc_antennae_r, avg_loc_antennae_r],
                                  [loc_leg_l_1, avg_loc_leg_l_1],
                                  [loc_leg_l_2, avg_loc_leg_l_2],
                                  [loc_leg_l_3, avg_loc_leg_l_3],
                                  [loc_leg_r_1, avg_loc_leg_r_1],
                                  [loc_leg_r_2, avg_loc_leg_r_2],
                                  [loc_leg_r_3, avg_loc_leg_r_3],
                                  [loc_abdomen, avg_loc_abdomen]]

                unit_vector_body_axis = [0, 1]

                for c, comb in enumerate(combined_poses):
                    point = [comb[1][0], comb[1][1]]
                    if centre_poses:
                        point = point - avg_front + [150, 120]

                    if align_poses:
                        point = rotate(point, [150, 120], degrees=body_orientation * 180 / np.pi)

                    bone_vector = [comb[1][0] - comb[0][0][0], comb[1][1] - comb[0][0][1]]
                    unit_vector_bone_vector = bone_vector / np.linalg.norm(bone_vector)
                    dot_product = np.dot(unit_vector_body_axis, unit_vector_bone_vector)
                    out_features_angles[frame * len(combined_poses) + c] = (180 / np.pi) * np.arccos(
                        np.clip(dot_product, -1.0, 1.0))

                    if display:
                        blank_image = cv2.circle(blank_image, (int(point[0]), int(point[1])),
                                                 5,
                                                 (255, 255, 255),
                                                 -1)

                if display:
                    cv2.imshow("test", blank_image)
                    cv2.waitKey(1)

            # get std for each point, if avg std is too high, discard, use weight means
            check_features = out_features.reshape(clip_length, num_features, 2)
            check_features_std = np.std(check_features, axis=0)
            mean_std = np.nanmean(check_features_std)

            if mean_std >= 45:
                skip_instance = True

            if skip_instance:
                continue

            if compute_size_class_for_each_action:
                x_diff = features[:, 0] - features[:, 15]  # b_tX -  b_a_5X
                y_diff = features[:, 1] - features[:, 16]  # b_tY -  b_a_5Y

                lengths = np.sqrt(np.square(x_diff) + np.square(y_diff))
                median_length = np.round(np.median(lengths) / px_to_mm_POSE, 2)

                # now, use the appropriate class ID
                size = find_class([3, 4, 5, 6, 7], median_length)

            # get speed from assocaited tracks
            temp_tracks = tracks[frame_start:frame_start + clip_length, track_id * 2 + 1: track_id * 2 + 3]
            temp_track_frames = np.arange(clip_length)
            temp_tracks = np.stack((temp_track_frames, temp_tracks[:, 0], temp_tracks[:, 1]), axis=1)
            tracks_MA = apply_moving_average(temp_tracks, 5)
            tracks_speed = get_derivative(tracks_MA, time_step=1 / 25)
            speed = max(np.nanmean(tracks_speed[np.nonzero(tracks_speed)], axis=0), 0.001) / px_to_mm_TRACK

            # get angular rates from computed angles
            temp_angle_frames = np.arange(clip_length)
            temp_angles = np.column_stack((temp_angle_frames, out_features_angles.reshape((clip_length, 9))))
            temp_angles_MA = apply_moving_average(temp_angles, 5)
            angular_velocity = get_derivative_angles(temp_angles_MA, time_step=1 / 25)
            angular_velocity = angular_velocity.flatten()

            # create list of all features, starting with static features
            temp_list = [os.path.basename(input_poses), material, frame_start, id_val, speed, size]

            # and extending the feature vector with dynamic features (pose, angles, angular rates)
            temp_list.extend(out_features.tolist())
            temp_list.extend(out_features_angles.tolist())
            temp_list.extend(angular_velocity.tolist())

            out_list.append(temp_list)

    cv2.destroyAllWindows()

    header = ["video", "material", "startframe", "id", "speed", "size_class"]
    feature_header = ["raw_pose_" + str(i) for i in range(int(num_features * 2 * clip_length))]
    feature_angle_header = ["angles_" + str(i) for i in range(9 * clip_length)]
    feature_angle_vel_header = ["angular_velocity_" + str(i) for i in range(9 * (clip_length - 1))]

    header.extend(feature_header)
    header.extend(feature_angle_header)
    header.extend(feature_angle_vel_header)

    df_ant_poses = pd.DataFrame(out_list)

    df_ant_poses.columns = header
    # fill NaNs with zeros to keep things clean
    df_ant_poses = df_ant_poses.fillna(0)

    out_df_file_name = str(os.path.basename(input_poses)) + "_POSE_FEATURE_DICT.pkl"
    if output_folder != "":
        out_path = os.path.join(output_folder, out_df_file_name)
    else:
        output_folder = out_df_file_name
    df_ant_poses.to_pickle(output_folder)
