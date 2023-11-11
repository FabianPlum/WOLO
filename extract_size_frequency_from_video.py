import cv2
import pickle
import argparse
import scipy.stats as stats
import os
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

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
                out_image[px_patch_y[0]:px_patch_y[1], px_patch_x[0]:px_patch_x[1]] = patch
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
    ap.add_argument("-l", "--min_track_length", default=100, required=False, type=int)
    ap.add_argument("-s", "--strip_tail_frames", default=50, required=False, type=int)
    ap.add_argument("-p", "--min_movement_px", default=50, required=False, type=int)
    ap.add_argument("-r", "--retrieve_every", default=5, required=False, type=int)
    ap.add_argument("-GPU", "--GPU", default="0", required=False, type=str)

    # optional
    ap.add_argument("-d", "--display_video", default=False, required=False, type=bool)

    args = vars(ap.parse_args())

    # set which GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["GPU"]

    """
    export_paths = ["C:/Users/Legos/OneDrive/EAEAAO/2019-07-23_bramble_leaves_right"]
    video = "C:/Users/Legos/OneDrive/EAEAAO/2019-07-23_bramble_leaves_right.avi"
    MODEL_PATH = "WEIGHT_ESTIMATOR/CLASS_MultiCamAnts-and-synth-standard_20_sigma-2_cross-entropy"
    """

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

    # get the resolution of the video file to automatically adjust the axis limits of the plot
    resolution = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    BATCH_SIZE = 128
    VERBOSE = 0
    LOSS = "categorical_crossentropy"
    OPTIMIZER = "adam"
    NUM_PARALLEL_CALLS = tf.data.AUTOTUNE
    MODEL_NAME = str(os.path.basename(MODEL_PATH))
    IMG_ROWS, IMG_COLS = 128, 128
    INPUT_SHAPE_RGB = (IMG_ROWS, IMG_COLS, 3)

    print(MODEL_NAME)

    if MODEL_NAME.split("_")[0] == "CLASS":
        if MODEL_NAME.split("_")[2] == "20":
            INFERENCE_METHOD = "CLASS_20"
            from model_training.TF2_SYNTH_CLASSIFICATION import *
            import model_training.TF2_SYNTH_CLASSIFICATION

            model_training.TF2_SYNTH_CLASSIFICATION.VERBOSE = VERBOSE
        else:
            INFERENCE_METHOD = "CLASS_5"
            from model_training.TF2_SYNTH_CLASSIFICATION_5_CLASS import *
            import model_training.TF2_SYNTH_CLASSIFICATION_5_CLASS

            model_training.TF2_SYNTH_CLASSIFICATION_5_CLASS.VERBOSE = VERBOSE
    else:
        INFERENCE_METHOD = "REG"
        from model_training.TF2_SYNTH_REGRESSION import *
        import model_training.TF2_SYNTH_REGRESSION

        model_training.TF2_SYNTH_REGRESSION.VERBOSE = VERBOSE

    print("INFO: Using trained model:", MODEL_NAME)
    print("INFO: Inference method:", INFERENCE_METHOD)

    if INFERENCE_METHOD.split("_")[0] == "CLASS":
        num_classes = int(MODEL_NAME.split("_")[2])
        print("\nINFO: Loading model...\n")
        model = build_with_Xception(input_shape=INPUT_SHAPE_RGB, output_nodes=num_classes)
        model.load_weights(os.path.join(MODEL_PATH, "cp-0050.ckpt"))

        model.compile(loss=LOSS,
                      optimizer=OPTIMIZER,
                      metrics=["accuracy", MAPE])  # [LOSS, "mean_squared_error", "mean_absolute_error"])
        model.summary()

        if num_classes == 5:
            model_training.TF2_SYNTH_CLASSIFICATION_5_CLASS.FIVE_CLASS = True

    if INFERENCE_METHOD == "REG":
        if MODEL_NAME.split("_")[-1] == "LOG":
            LOSS = MODEL_NAME.split("_")[-2]
            LOG = True
            model_training.TF2_SYNTH_REGRESSION.LOG = True
        else:
            LOSS = MODEL_NAME.split("_")[-1]
            LOG = False
            model_training.TF2_SYNTH_REGRESSION.LOG = False

        print("\nINFO: Loading model...\n")
        model = build_with_Xception(input_shape=INPUT_SHAPE_RGB, output_nodes=1)
        model.load_weights(os.path.join(MODEL_PATH, "cp-0050.ckpt"))

        model.compile(loss=LOSS,
                      optimizer=OPTIMIZER,
                      metrics=[MAPE])  # [LOSS, "mean_squared_error", "mean_absolute_error"])
        model.summary()

    num_tracks = int((np.array(tracks[0]).shape[1] - 1) / 2)
    num_frames = tracked_frames - strip_tail_frames

    # create array holding all possible weight measurements
    all_weights = np.full([math.floor(num_frames / retrieve_every), num_tracks], np.nan)

    tracks_np = np.array(tracks[0])

    # Iterate over all frames, retrieve stacks of valid patches, run inference, and add their values to all_weights
    # It is likely MUCH faster to simply iterate over all frames in the video and every n-th frame extract and predict
    for frame_num in tqdm(range(0, tracked_frames - 2 * strip_tail_frames)):
    #for frame_num in range(0, tracked_frames - 2 * strip_tail_frames):
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

            # now pass patch_stack to model.predict(patch_stack) and return predicted labels to assign to all_weights
            predictions = model.predict(patch_stack, verbose=0)

            # convert predictions to their final outputs, depending on the inference method
            if INFERENCE_METHOD == "CLASS_5":
                # get predicted class from one_hot encoded prediction vector
                y_pred_class = np.argmax(predictions, axis=1)

                class_list = [0.0013, 0.0030, 0.0068, 0.0154, 0.0351]

                out_predictions = np.take(class_list, np.argmax(predictions, axis=1))

            elif INFERENCE_METHOD == "CLASS_20":
                # get predicted class from one_hot encoded prediction vector
                y_pred_class = np.argmax(predictions, axis=1)

                class_list = [0.0010, 0.0012, 0.0015, 0.0019, 0.0023,
                              0.0028, 0.0034, 0.0042, 0.0052, 0.0064,
                              0.0078, 0.0096, 0.0118, 0.0145, 0.0179,
                              0.0219, 0.0270, 0.0331, 0.0407, 0.0500]

                out_predictions = np.take(class_list, np.argmax(predictions, axis=1))

            elif INFERENCE_METHOD == "REG":
                if LOG:
                    predictions = delog_and_denorm(predictions)
                    out_predictions = predictions.numpy().reshape(tf.shape(predictions)[0])
                else:
                    out_predictions = predictions.reshape(len(predictions))

            for patch_id, pred in zip(patch_ids, out_predictions):
                all_weights[math.floor(frame_num / retrieve_every), patch_id] = pred

    # export all_tracks and all_weights
    with open(str(os.path.basename(video))[:-4] + "_ALL_CLEANED_TRACKS.pickle", 'wb') as handle:
        pickle.dump(tracks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # export all_tracks and all_weights
    with open(str(os.path.basename(video))[:-4] + "_ALL_WEIGHTS.pickle", 'wb') as handle:
        pickle.dump(all_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_weights_compressed_mean = np.round(np.nanmean(all_weights, axis=0), 4)
    all_weights_compressed_median = np.round(np.nanmedian(all_weights, axis=0), 4)
    all_weights_compressed_mode = np.round(stats.mode(all_weights, axis=0, nan_policy='omit', keepdims=True)[0][0], 4)

    extracted_weights = all_weights_compressed_mean.shape
    print("\nFinal weight estimates for", extracted_weights, "individuals:")
    print(all_weights_compressed_mean)
    print("\n")
    print(all_weights_compressed_median)
    print("\n")
    print(all_weights_compressed_mode)
    print("\n")

    # regardless of inference type, create a 20-bin size-frequency distribution plot
    twenty_class = [0.0010, 0.0012, 0.0015, 0.0019, 0.0023, 0.0028,
                    0.0034, 0.0042, 0.0052, 0.0064, 0.0078, 0.0096,
                    0.0118, 0.0145, 0.0179, 0.0219, 0.0270, 0.0331,
                    0.0407, 0.0500]

    """
    
    we extract the following plots with mean, median, and mode of the predicted weights per track
    
    """
    # produced using MEAN
    pred_classes = [find_class(twenty_class, float(x)) for x in all_weights_compressed_mean]
    pred_classes = clean_array(pred_classes, strip_NaN=True, strip_zero=True)

    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

    # Plot Histogram on x
    fig, ax = plt.subplots()
    ax.hist(pred_classes, bins=range(20), density=True)
    ax.set_xticks(np.arange(len(twenty_class)))
    ax.set_xticklabels(twenty_class, rotation=45)
    ax.set_ylim(0, 1)
    plt.gca().set(title='size-frequency distribution', ylabel='relative frequency')

    print("Size-frequency distribution (mean) plot produced for", os.path.basename(video),
          "containing n =", len(all_weights_compressed_mean), "valid tracks.")

    plt.savefig(
        "Size-frequency distribution (mean) plot produced for " + str(os.path.basename(video))[:-4] + "_mean.svg")

    # produced using MEDIAN
    pred_classes = [find_class(twenty_class, float(x)) for x in all_weights_compressed_median]
    pred_classes = clean_array(pred_classes, strip_NaN=True, strip_zero=True)

    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

    # Plot Histogram on x
    fig, ax = plt.subplots()
    ax.hist(pred_classes, bins=range(20), density=True)
    ax.set_xticks(np.arange(len(twenty_class)))
    ax.set_xticklabels(twenty_class, rotation=45)
    ax.set_ylim(0, 1)
    plt.gca().set(title='size-frequency distribution', ylabel='relative frequency')

    print("Size-frequency distribution (median) plot produced for", os.path.basename(video),
          "containing n =", len(all_weights_compressed_mode), "valid tracks.")

    plt.savefig(
        "Size-frequency distribution (median) plot produced for " + str(os.path.basename(video))[:-4] + "_median.svg")

    # produced using MODE
    pred_classes = [find_class(twenty_class, float(x)) for x in all_weights_compressed_mode]
    pred_classes = clean_array(pred_classes, strip_NaN=True, strip_zero=True)

    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

    # Plot Histogram on x
    fig, ax = plt.subplots()
    ax.hist(pred_classes, bins=range(20), density=True)
    ax.set_xticks(np.arange(len(twenty_class)))
    ax.set_xticklabels(twenty_class, rotation=45)
    ax.set_ylim(0, 1)
    plt.gca().set(title='size-frequency distribution', ylabel='relative frequency')

    print("Size-frequency distribution (mode) plot produced for", os.path.basename(video),
          "containing n =", len(all_weights_compressed_mode), "valid tracks.")

    plt.savefig(
        "Size-frequency distribution (mode) plot produced for " + str(os.path.basename(video))[:-4] + "_mode.svg")
