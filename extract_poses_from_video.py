import cv2
import pickle
import argparse
import scipy.stats as stats
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from dlclive import DLCLive, Processor
import yaml
import numpy as np

# now for the file management functions
from evaluation.Antrax_base import import_tracks, display_video


def get_patch_stack(frame, tracks, frame_num=0, patch_size=128, show_patches=False, VERBOSE=False):
    # create an empty array to which we append all extracted patches
    patches = []
    patch_ids = []

    # then retrieve all track centres at the given frame
    track_centres = tracks[frame_num, 1:]
    track_centres_non_zero_ids = np.nonzero(track_centres)[0][::2]

    if VERBOSE:
        print("Valid tracks at", frame_num)
        print(track_centres_non_zero_ids)

    if len(track_centres_non_zero_ids) != 0:

        for track_orig in track_centres_non_zero_ids:
            track = int((track_orig + 1) / 2)
            # invert y-axis, to fit openCV convention ( lower left -> (x=0,y=0) )
            target_centre = [track_centres[track * 2], frame.shape[0] - track_centres[track * 2 + 1]]
            # define the starting and ending point of the bounding box rectangle, defined by "target_size"
            px_start = target_centre - np.asarray([math.floor(patch_size / 2), math.floor(patch_size / 2)])
            min_frame = [np.max([px_start[0], 0]), np.max([px_start[1], 0])]
            px_end = target_centre + np.asarray([math.floor(patch_size / 2), math.floor(patch_size / 2)])
            max_frame = [np.min([px_end[0], frame.shape[1]]), np.min([px_end[1], frame.shape[0]])]
            # extract the defined rectangle of the track from the frame and save to the stack
            patch = frame[min_frame[1]:max_frame[1], min_frame[0]:max_frame[0]]
            if patch.shape != (patch_size, patch_size, 3):
                # start out with a blank image
                out_image = np.zeros((patch_size, patch_size, 3), np.uint8)
                # now insert the patch into the centre of the blank image
                px_patch_y = [math.floor((patch_size - patch.shape[0]) / 2),
                              math.floor((patch_size - patch.shape[0]) / 2) + patch.shape[0]]
                px_patch_x = [math.floor((patch_size - patch.shape[1]) / 2),
                              math.floor((patch_size - patch.shape[1]) / 2) + patch.shape[1]]
                try:
                    out_image[px_patch_y[0]:px_patch_y[1], px_patch_x[0]:px_patch_x[1]] = patch
                except ValueError:
                    print("WARNING: Incorrectly sized patch found! Skipping:", track_orig, "at frame", frame_num)
                    continue
                else:
                    out_image = patch
                if show_patches:
                    cv2.imshow(str(track), out_image)
                    cv2.waitKey(0)

                patches.append(out_image)
                patch_ids.append(track)

        patch_stack = np.array(patches)
        if VERBOSE:
            print("Stack shape:", patch_stack.shape)

        if show_patches:
            cv2.destroyAllWindows()

        return patch_stack, patch_ids

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

    if __name__ == "__main__":

        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        # Data input and output
        ap.add_argument("-t", "--tracks", required=True, type=str)
        ap.add_argument("-v", "--video", required=True, type=str)
        ap.add_argument("-m", "--MODEL_PATH", required=True, type=str)

        # processing parameters
        ap.add_argument("-tr", "--threshold", default=0.5, required=False, type=float)
        ap.add_argument("-l", "--min_track_length", default=100, required=False, type=int)
        ap.add_argument("-s", "--strip_tail_frames", default=50, required=False, type=int)
        ap.add_argument("-p", "--min_movement_px", default=50, required=False, type=int)
        ap.add_argument("-r", "--retrieve_every", default=1, required=False, type=int)
        ap.add_argument("-GPU", "--GPU", default=None, required=False, type=str)

        # optional
        ap.add_argument("-d", "--display_video", default=False, required=False, type=bool)

        args = vars(ap.parse_args())

        # set which GPU to use
        if args["GPU"] is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = args["GPU"]

        export_paths = [args["tracks"]]
        video = args["video"]
        MODEL_PATH = args["MODEL_PATH"]

        # enter the number of annotated frames:
        min_track_length = int(args["min_track_length"])  # set to 0 if all tracks should be used, regardless of length
        strip_tail_frames = int(args["strip_tail_frames"])  # set to zero if all are kept (used to strip extrapolation)
        min_movement_px = int(args["min_movement_px"])  # controls how "still" a subject may be before its excluded
        retrieve_every = int(args["retrieve_every"])  # every how many frames estimate is to be performed for each track

        # now we can load the captured video file and display it
        cap = cv2.VideoCapture(video)

        # check the number of frames of the imported video file
        numFramesMax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("The imported clip:", video, "has a total of", numFramesMax, "frames.\n")

        # now let's load all tracks listed in the export_paths
        tracks = []
        for folder in export_paths:
            # You can export all tracks into a single .csv file by setting "export=True"
            tracks.append(import_tracks(folder, numFramesMax,
                                        export=False,
                                        min_track_length=min_track_length,
                                        strip_tail_frames=strip_tail_frames,
                                        min_movement_px=min_movement_px))

            # The following function is used to display the tracks you imported.
            # You can press "q" while hovering over the displayed video to exit.
            print("\nDisplaying tracks loaded from:", folder)
            tracked_frames = len(tracks[0])
            if args["display_video"]:
                display_video(cap, tracks[-1], show=(0, tracked_frames - strip_tail_frames), scale=0.6)

        tracks_np = np.array(tracks[0])

        # get the resolution of the video file to automatically adjust the axis limits of the plot
        resolution = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

        dlc_proc = Processor()
        print("Loading DLC Network from", MODEL_PATH)
        dlc_live = DLCLive(MODEL_PATH, processor=dlc_proc, pcutoff=args["threshold"])

        # create a list of join names from those defined in the pose_cfg.yaml file
        dlc_pose_cfg = os.path.join(MODEL_PATH, "pose_cfg.yaml")
        with open(dlc_pose_cfg, "r") as stream:
            pose_cfg_yaml = yaml.safe_load(stream)

        pose_joint_names = pose_cfg_yaml["all_joints_names"]
        pose_joint_header_l1 = "scorer," + ",".join("OmniTrax,OmniTrax,OmniTrax" for e in pose_joint_names)
        pose_joint_header_l2 = "bodyparts," + ",".join(str(e) + "," +
                                                       str(e) + "," +
                                                       str(e) for e in pose_joint_names)
        pose_joint_header_l3 = "coords," + ",".join("x," +
                                                    "y," +
                                                    "likelihood" for e in pose_joint_names)

        dlc_config_path = os.path.join(MODEL_PATH, "config.yaml")
        with open(dlc_config_path, "r") as stream:
            config_yaml = yaml.safe_load(stream)
            print("skeleton configuration:\n", config_yaml["skeleton"])

            # now, match the skeleton elements to their IDs to draw them as overlays
            skeleton = []
            try:
                for bone in config_yaml["skeleton"]:
                    skeleton.append([pose_joint_names.index(bone[0]),
                                     pose_joint_names.index(bone[1])])

                print("skeleton links:\n", skeleton)
            except ValueError:
                print("Your config skeleton and pose joint names do not match!"
                      "\n could not create overlay skeleton!")
                skeleton = []

        MODEL_NAME = str(os.path.basename(MODEL_PATH))
        print("INFO: Using trained model:", MODEL_NAME)

        num_tracks = int((np.array(tracks[0]).shape[1] - 1) / 2)
        num_frames = tracked_frames - strip_tail_frames

        network_initialised = False

        track_poses = [{} for i in range(num_tracks)]

        # Iterate over all frames, retrieve stacks of valid patches, run inference, and add their values to all_weights
        # It is likely MUCH faster to simply iterate over all frames in the video and every n-th frame extract and predict
        for frame_num in tqdm(range(0, tracked_frames - 2 * strip_tail_frames)):
            # for frame_num in range(0, tracked_frames - 2 * strip_tail_frames):
            ret, frame = cap.read()
            if not ret:
                break
                print("WARNING: Video has ended!")
            if ret and frame_num % retrieve_every == 0:
                patch_stack, patch_ids = get_patch_stack(frame, tracks_np, frame_num=frame_num, patch_size=128,
                                                         show_patches=False, VERBOSE=False)

                # skip empty frames
                if len(patch_ids) == 0:
                    continue

                # initialise network (if it has not been initialised yet)
                if not network_initialised:
                    dlc_live.init_inference(patch_stack[0])
                    network_initialised = True

                for pID, patch in zip(patch_ids, patch_stack):

                    dlc_input_img = cv2.resize(patch.copy(), (300, 300))
                    # estimate pose in cropped frame
                    pose = dlc_live.get_pose(dlc_input_img)
                    track_poses[pID][str(frame_num)] = pose.flatten()

                    if args["display_video"]:
                        for p, point in enumerate(pose):
                            if point[2] >= args["threshold"]:
                                dlc_input_img = cv2.circle(dlc_input_img, (int(point[0]), int(point[1])),
                                                           3,
                                                           (int(255 * p / len(pose_joint_names)),
                                                            int(255 - 255 * p / len(pose_joint_names)), 200),
                                                           -1)

                        for b, bone in enumerate(skeleton):
                            if pose[bone[0]][2] >= args["threshold"] and pose[bone[1]][2] >= args["threshold"]:
                                dlc_input_img = cv2.line(dlc_input_img,
                                                         (int(pose[bone[0]][0]), int(pose[bone[0]][1])),
                                                         (int(pose[bone[1]][0]), int(pose[bone[1]][1])),
                                                         (120, 220, 120),
                                                         1)

                        cv2.imshow("DLC Pose Estimation", dlc_input_img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

        out_dir = str(os.path.basename(args["video"]))[:-4]

        if not os.path.exists(out_dir):
            print("INFO: Created output directory for paths:", out_dir)
            os.makedirs(out_dir)

        for pID, track in enumerate(track_poses):
            if not track:
                continue
            pose_output_file = open(
                os.path.join(out_dir, str(os.path.basename(args["video"]))[:-4] + "_POSE_" + str(pID) + ".csv"), "w")
            # write header line
            # pose_output_file.write("frame," + pose_joint_header + ",r1_deg,r2_deg,r3_deg,l1_deg,l2_deg,l3_deg\n")
            """
            replicate DLC prediction output file structure:
            scorer    | OmniTrax  | OmniTrax  | OmniTrax   | ...
            bodyparts | part_A    | part_A    | part_A     | ...
            coords    | x         | y         | likelihood | ...
            """
            # write header line
            pose_output_file.write(pose_joint_header_l1 + "\n")
            pose_output_file.write(pose_joint_header_l2 + "\n")
            pose_output_file.write(pose_joint_header_l3 + "\n")
            for key, value in track.items():
                line = key + "," + ",".join(str(e) for e in value.flatten())
                pose_output_file.write(line + "\n")
            pose_output_file.close()
