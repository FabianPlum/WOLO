import os  # to perform file operations
from os.path import isfile, join
import argparse
import pickle
import math
import random
from sklearn.utils import shuffle
from contextlib import redirect_stdout

# limit number of simultaneously used cores / threads
# os.environ["OPENBLAS_NUM_THREADS"] = "8"  # export OPENBLAS_NUM_THREADS=8

# set cuda visible devices when executing locally on a multi-GPU system
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow import keras
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)


# import function to load image data and read labels for the y vector
def importLabeledImages(target_directory, make_five_class=True):
    X = []
    y = []
    y_gt = []
    unique_classes = 0
    y_five_class_list = [0.0013, 0.0030, 0.0068, 0.0154, 0.0351]
    class_counter = -1

    for cl, folder in enumerate(os.listdir(target_directory)):
        print(join(target_directory, folder))
        if make_five_class:
            if cl % 4 == 0:
                class_counter += 1
            print("Adjusted class:", class_counter, "corresponding to", y_five_class_list[class_counter])
        for filename in os.listdir(join(target_directory, folder)):
            X.append(join(target_directory, folder, filename))
            if make_five_class:
                y.append(class_counter)
            else:
                y.append(cl)
            try:
                y_val = float(filename.split(" ")[-1][:-4])
            except ValueError:
                try:
                    y_val = float(filename.split("_")[-1][:-4])
                except ValueError:
                    y_val = float(filename.split("_")[-1][:-4].split("-")[-1])

            if y_val > 1:
                y_val /= 10000  # so all values are treated as grams
                """
                currently some datasets use 0.003 to express 3 mg
                and some use 00030 to express 3 mg
                """
            y_gt.append(y_val)

        if not make_five_class:
            unique_classes += 1

    if make_five_class:
        unique_classes = int(class_counter + 1)

    return X, y, y_gt, unique_classes


def MAPE(y_true, y_pred, classes=[0.0010, 0.0012, 0.0015, 0.0019, 0.0023, 0.0028,
                                  0.0034, 0.0042, 0.0052, 0.0064, 0.0078, 0.0096,
                                  0.0118, 0.0145, 0.0179, 0.0219, 0.0270, 0.0331,
                                  0.0407, 0.0500],
         y_five_class_list=[0.0013, 0.0030, 0.0068, 0.0154, 0.0351]):
    """
    Compute the MAPE (Mean Absolute Percentage Error) from the true and the predicted classes
    Supports batches, thus shapes of (batch x num_classes)
    :param y_true: ground truth of length m x n
    :param y_pred: prediction of length m x n
    :param classes: int or float vector of length n
    :return: MAPE
    """

    if FIVE_CLASS:
        classes = y_five_class_list

    try:
        if y_true.shape[0] is None:
            batchsize = 1
        else:
            batchsize = y_true.shape[0]
    except AttributeError:
        batchsize = len(y_true)

    MAPE_list = []

    for b in range(batchsize):
        y_true_class = tf.gather(classes, tf.argmax(y_true[b]))  # classes[tf.argmax(y_true[b], axis=1)]
        y_pred_class = tf.gather(classes, tf.argmax(y_pred[b]))  # classes[tf.argmax(y_pred[b], axis=1)]

        APE = tf.math.abs(tf.math.divide((y_true_class - y_pred_class), y_true_class))
        MAPE_list.append(APE)

    MAPE = tf.math.multiply(tf.constant(100, tf.float32), tf.math.reduce_mean(tf.stack(MAPE_list)))
    return MAPE


def MAPE_true(y_gt, y_pred, classes=[0.0010, 0.0012, 0.0015, 0.0019, 0.0023, 0.0028,
                                     0.0034, 0.0042, 0.0052, 0.0064, 0.0078, 0.0096,
                                     0.0118, 0.0145, 0.0179, 0.0219, 0.0270, 0.0331,
                                     0.0407, 0.0500],
              classes_upper_limit=[0.0011, 0.0013, 0.0017, 0.0021, 0.0025, 0.0031,
                                   0.0038, 0.0047, 0.0058, 0.0071, 0.0087, 0.0107,
                                   0.0131, 0.0162, 0.0199, 0.0244, 0.0300, 0.0369, 0.0453
                                   ],
              y_five_class_list=[0.0013, 0.0030, 0.0068, 0.0154, 0.0351],
              y_five_class_list_upper_limit=[0.0021, 0.0047, 0.0107, 0.0244]):
    """
    Compute the MAPE_true (Mean Absolute Percentage Error) from the ground truth value and the predicted classes
    Supports batches, thus shapes of (batch x num_classes)
    :param y_gt: ground truth of length m
    :param y_pred: prediction of length m x n
    :param classes: int or float vector of length n
    :param classes_upper_limit: int or float vector of length n, containing upper class limit to compute MAPE_ideal
    :return: MAPE_true, MAPE_ideal
    """

    if FIVE_CLASS:
        classes = y_five_class_list
        classes_upper_limit=y_five_class_list_upper_limit

    if y_pred.shape[0] is None:
        batchsize = 1
    else:
        batchsize = y_pred.shape[0]

    MAPE_list = []
    MAPE_ideal_list = []

    for b in range(batchsize):
        y_gt_val = y_gt[b]
        y_class_temp = 0
        for c, cl in enumerate(classes_upper_limit):
            if cl < y_gt_val:
                y_class_temp = c
            else:
                break

        y_pred_class = tf.gather(classes, tf.argmax(y_pred[b]))  # classes[tf.argmax(y_pred[b], axis=1)]
        # get APE_true
        APE_true = tf.math.abs(tf.math.divide((y_gt_val - y_pred_class), y_gt_val))
        MAPE_list.append(APE_true)

        # get APE_ideal
        APE_idel = tf.math.abs(tf.math.divide((y_gt_val - classes_upper_limit[y_class_temp]), y_gt_val))
        MAPE_ideal_list.append(APE_idel)

    MAPE_true_val = tf.math.multiply(tf.constant(100, tf.float32), tf.math.reduce_mean(tf.stack(MAPE_list)))
    MAPE_ideal_val = tf.math.multiply(tf.constant(100, tf.float32), tf.math.reduce_mean(tf.stack(MAPE_ideal_list)))
    return MAPE_true_val, MAPE_ideal_val


def make_gauss_pdf(y, sigma=1, sum_to_one=True, num_classes=20):
    """
    y : label vector (one-hot encoded) with length n classes
    sigma : standard deviation of probability density function

    the distribution will be centered around the maximum activation
    which will subsequently treated as the dsitribution mean
    """

    y_gauss = []

    # mu = 3  #
    if y.shape == () or y.shape[0] is None:
        mu = tf.constant(0)
    else:
        mu = tf.argmax(y)

    for c in range(num_classes):
        y_gauss_a = tf.math.divide(tf.constant(1, tf.float32),
                                   tf.math.multiply(tf.constant(sigma, tf.float32),
                                                    tf.math.sqrt(
                                                        tf.constant(2 * math.pi, tf.float32)))),
        y_gauss_b = tf.math.exp(
            tf.math.divide(-tf.math.square(tf.constant(c, tf.float32) - tf.cast(mu, tf.float32)),
                           tf.math.multiply(tf.constant(2, tf.float32),
                                            tf.math.square(tf.constant(sigma, tf.float32)))))

        y_gauss.append(tf.math.multiply(y_gauss_a, y_gauss_b))

    # to use softmax for the final activation layer the sum of the PDF should be equal to one
    y_gauss = tf.stack(y_gauss)

    if sum_to_one:
        mult_fact = tf.math.reduce_sum(y_gauss)
    y_gauss = tf.math.divide(y_gauss, mult_fact)

    return tf.reshape(y_gauss, y.shape)


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


def balance_classes(X, y, y_gt, shuffle=False, SEED=0):
    # assumes values are shuffled, if not set shuffle to True
    if shuffle:
        X, y, y_gt = shuffle(X, y, y_gt, trandom_state=SEED)

    classes, classes_count = np.unique(np.array(y), return_counts=True)

    ordered_class_representation = classes_count.argsort()
    classes = classes[ordered_class_representation]
    max_num_samples = classes_count[ordered_class_representation][0]

    print("Minimum num samples:", classes_count[ordered_class_representation][0], "for class", classes[0])
    print("Reducing number of samples for all other classes to balance dataset to smallest representation...")

    X_new, y_new, y_gt_new = [], [], []

    element_counter = np.zeros(len(classes))

    for x_elem, y_elem, y_gt_elem in zip(X, y, y_gt):
        if element_counter[y_elem] < max_num_samples:
            X_new.append(x_elem)
            y_new.append(y_elem)
            y_gt_new.append(y_gt_elem)
            element_counter[y_elem] += 1

    return X_new, y_new, y_gt_new


def build_with_Xception(input_shape, output_nodes=1, refine=False):
    # Load weights pre-trained on ImageNet.
    # Do not include the ImageNet classifier at the top.
    base_model = keras.applications.Xception(weights="imagenet",
                                             input_shape=input_shape,
                                             include_top=False)

    # by default, freeze the weights of the backbone
    # Importantly, when the base model is set trainable, it is still running in
    # inference mode since we passed training=False (see below xa = ...) when calling it.
    # This means that the batch normalization layers inside won't update their batch statistics.
    # If they did, they would wreck havoc on the representations learned by the model so far.
    base_model.trainable = refine

    inputs = keras.Input(shape=input_shape)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    xa = base_model(x, training=refine)
    x1 = keras.layers.GlobalAveragePooling2D()(xa)
    x1_d = tf.keras.layers.Dropout(.2)(x1)
    # A Dense classifier with a single unit
    x2 = keras.layers.Dense(4096, activation="relu")(x1_d)  # 1024
    x2_d = tf.keras.layers.Dropout(.2)(x2)

    x3 = keras.layers.Dense(4096, activation="relu")(x2_d)  # 1024
    x3_d = tf.keras.layers.Dropout(.2)(x3)

    # x4 = keras.layers.Dense(1024, activation="relu")(x3_d)  # 1024
    # x4_d = tf.keras.layers.Dropout(.2)(x4)

    outputs = keras.layers.Dense(output_nodes, activation='softmax')(x3_d)

    model = keras.Model(inputs, outputs)

    return model


def build_with_EfficientNet(input_shape, output_nodes=1, refine=False):
    # Load weights pre-trained on ImageNet.
    # Do not include the ImageNet classifier at the top.
    base_model = keras.applications.efficientnet.EfficientNetB7(weights="imagenet",
                                                                input_shape=input_shape,
                                                                include_top=False)
    # freeze the weights of the backbone
    base_model.trainable = refine

    inputs = keras.Input(shape=input_shape)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    # x = scale_layer(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    xa = base_model(inputs, training=False)

    x1 = keras.layers.GlobalAveragePooling2D()(xa)
    x1_d = tf.keras.layers.Dropout(.2)(x1)
    # A Dense classifier with a single unit
    x2 = keras.layers.Dense(4096, activation="relu")(x1_d)  # 1024
    x2_d = tf.keras.layers.Dropout(.2)(x2)

    x3 = keras.layers.Dense(4096, activation="relu")(x2_d)  # 1024
    x3_d = tf.keras.layers.Dropout(.2)(x3)

    outputs = keras.layers.Dense(output_nodes, activation='softmax')(x3_d)

    model = keras.Model(inputs, outputs)

    return model


def build_simple_cnn(input_shape, output_nodes=1):
    """
    Build a simple convolutional neural network with standard regularization techniques
    """
    inputs = keras.Input(shape=input_shape, name='input_layer')
    
    # First convolutional block
    conv1_1 = keras.layers.Conv2D(32, (3, 3), padding='same', name='conv1_1')(inputs)
    bn1_1 = keras.layers.BatchNormalization(name='bn1_1')(conv1_1)
    act1_1 = keras.layers.Activation('relu', name='relu1_1')(bn1_1)
    conv1_2 = keras.layers.Conv2D(32, (3, 3), padding='same', name='conv1_2')(act1_1)
    bn1_2 = keras.layers.BatchNormalization(name='bn1_2')(conv1_2)
    act1_2 = keras.layers.Activation('relu', name='relu1_2')(bn1_2)
    pool1 = keras.layers.MaxPooling2D((2, 2), name='pool1')(act1_2)
    drop1 = keras.layers.Dropout(0.25, name='dropout1')(pool1)

    # Second convolutional block
    conv2_1 = keras.layers.Conv2D(64, (3, 3), padding='same', name='conv2_1')(drop1)
    bn2_1 = keras.layers.BatchNormalization(name='bn2_1')(conv2_1)
    act2_1 = keras.layers.Activation('relu', name='relu2_1')(bn2_1)
    conv2_2 = keras.layers.Conv2D(64, (3, 3), padding='same', name='conv2_2')(act2_1)
    bn2_2 = keras.layers.BatchNormalization(name='bn2_2')(conv2_2)
    act2_2 = keras.layers.Activation('relu', name='relu2_2')(bn2_2)
    pool2 = keras.layers.MaxPooling2D((2, 2), name='pool2')(act2_2)
    drop2 = keras.layers.Dropout(0.25, name='dropout2')(pool2)

    # Third convolutional block
    conv3_1 = keras.layers.Conv2D(128, (3, 3), padding='same', name='conv3_1')(drop2)
    bn3_1 = keras.layers.BatchNormalization(name='bn3_1')(conv3_1)
    act3_1 = keras.layers.Activation('relu', name='relu3_1')(bn3_1)
    conv3_2 = keras.layers.Conv2D(128, (3, 3), padding='same', name='conv3_2')(act3_1)
    bn3_2 = keras.layers.BatchNormalization(name='bn3_2')(conv3_2)
    act3_2 = keras.layers.Activation('relu', name='relu3_2')(bn3_2)
    pool3 = keras.layers.MaxPooling2D((2, 2), name='pool3')(act3_2)
    drop3 = keras.layers.Dropout(0.25, name='dropout3')(pool3)

    # Dense layers
    flatten = keras.layers.Flatten(name='flatten')(drop3)
    dense1 = keras.layers.Dense(512, name='dense1')(flatten)
    bn_dense1 = keras.layers.BatchNormalization(name='bn_dense1')(dense1)
    act_dense1 = keras.layers.Activation('relu', name='relu_dense1')(bn_dense1)
    drop_dense1 = keras.layers.Dropout(0.5, name='dropout_dense1')(act_dense1)
    
    outputs = keras.layers.Dense(output_nodes, activation='softmax', name='output')(drop_dense1)

    model = keras.Model(inputs, outputs, name='simple_cnn')
    return model


if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-s", "--sigma", default=0, type=float)
    ap.add_argument("-d", "--dataset", required=True, type=str)
    ap.add_argument("-r", "--rand_seed", default=0, required=False, type=int)
    ap.add_argument("-o", "--output_dir", required=True, type=str)
    ap.add_argument("-b", "--backbone", default=None, required=False, type=str)

    # Optional flags
    ap.add_argument("-ba", "--balance_classes", default=False, required=False, type=bool)
    ap.add_argument("-e", "--epochs", default=10, required=False, type=int)
    ap.add_argument("-bs", "--batch_size", default=128, required=False, type=int)
    ap.add_argument("-sw", "--save_weights_every", default=10, required=False, type=int)
    ap.add_argument("-aug", "--augmentation", default=False, required=False, type=bool)
    ap.add_argument("-t", "--test", default=False, required=False, type=str)
    ap.add_argument("-fc", "--five_class", default=True, required=False, type=bool)
    ap.add_argument("-lr", "--learning_rate", default=0.0001, required=False, type=float)
    ap.add_argument("-rX", "--refine_Xception", default=False, required=False, type=bool)
    ap.add_argument("-cp", "--checkpoint", default=None, required=False, type=str,
                    help="Path to checkpoint to resume training from")
    
    args = vars(ap.parse_args())

    # Load the TensorBoard notebook extension
    # %load_ext tensorboard

    EPOCHS = int(args["epochs"])
    SAVE_WEIGHTS_EVERY = int(args["save_weights_every"])
    BATCH_SIZE = int(args["batch_size"])
    VERBOSE = 2
    SIGMA = float(args["sigma"])  # for class-aware gaussian label smoothing
    OPTIMIZER = "adam"
    LOSS = "categorical_crossentropy"
    IMG_ROWS, IMG_COLS = 128, 128
    INPUT_SHAPE_RGB = (IMG_ROWS, IMG_COLS, 3)
    NUM_PARALLEL_CALLS = tf.data.AUTOTUNE
    DATA_PATH = args["dataset"]
    TEST_DATA = args["test"]
    SEED = int(args["rand_seed"])
    AUGMENTATION = args["augmentation"]
    LEARNING_RATE = float(args["learning_rate"])

    print("\n--------------------------------------",
          "\nINFO: Training Settings\n",
          "\nEPOCHS: ", EPOCHS,
          "\nSAVE_WEIGHTS_EVERY: ", SAVE_WEIGHTS_EVERY,
          "\nBATCH_SIZE: ", BATCH_SIZE,
          "\nVERBOSE: ", VERBOSE,
          "\nSIGMA: ", SIGMA,
          "\nOPTIMIZER: ", OPTIMIZER,
          "\nLEARNING_RATE: ", LEARNING_RATE,
          "\nLOSS: ", LOSS,
          "\nDATA_PATH: ", DATA_PATH,
          "\nTEST_DATA: ", TEST_DATA,
          "\nAUGMENTATION: ", AUGMENTATION,
          "\nSEED: ", SEED)

    if args["balance_classes"]:
        print("\nINFO: Using balanced classes (all classes contain number of samples of smallest class)")
    else:
        print("\nINFO: Classes are not re-balanced")

    if SIGMA > 0:
        print("INFO: Using class-aware gaussian label smoothing")
    else:
        print("INFO: Using default one-hot encoding")

    if args["augmentation"] == True:
        print("\nINFO: Data augmentation - enabled\n")
    else:
        print("\nINFO: Data augmentation - disabled\n")

    print("\n--------------------------------------\n")

    # set all randomisation seeds for reproducible results
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    FIVE_CLASS = args["five_class"]

    # use CPU for all pre-processing
    with tf.device('/cpu:0'):
        # load paths
        X_paths, y_paths, y_gt, NUM_CLASSES = importLabeledImages(DATA_PATH,
                                                                  make_five_class=FIVE_CLASS)

        # 1 shuffle
        X_paths, y_paths, y_gt = shuffle(X_paths, y_paths, y_gt, random_state=SEED)

        # 1.5 balance classes

        if args["balance_classes"] == True:
            X_paths, y_paths, y_gt = balance_classes(X_paths, y_paths, y_gt)

        train_ds = tf.data.Dataset.from_tensor_slices((X_paths, y_paths))
        # normally shuffle should be equal to len(filenames), but as we shuffle in the preprocessing step
        # BATCH_SIZE * 100 is acceptable here
        train_ds = train_ds.shuffle(BATCH_SIZE * 100)
        train_ds = train_ds.map(parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)
        train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)),
                                num_parallel_calls=NUM_PARALLEL_CALLS)

        # using one-hot encoding with class-aware gaussian label smoothing if SIGMA is > 0
        if SIGMA > 0:
            train_ds = train_ds.map(lambda x, y: (x, make_gauss_pdf(y,
                                                                    sigma=SIGMA,
                                                                    sum_to_one=True,
                                                                    num_classes=NUM_CLASSES)),
                                    num_parallel_calls=NUM_PARALLEL_CALLS)

        # apply data augmentation, if enabled
        if args["augmentation"] == True:
            trainAug = tf.keras.Sequential([
                # Spatial augmentations
                tf.keras.layers.RandomFlip("horizontal", seed=SEED),
                tf.keras.layers.RandomRotation(
                    factor=0.1,  # ±10% rotation
                    seed=SEED,
                    fill_mode='reflect'
                ),
                tf.keras.layers.RandomZoom(
                    height_factor=(-0.1, 0.1),  # Allow both zoom in and out
                    width_factor=(-0.1, 0.1),
                    seed=SEED,
                    fill_mode='reflect'
                ),
                
                # Intensity augmentations - more conservative
                tf.keras.layers.RandomBrightness(
                    factor=0.05,  # ±5% brightness
                    seed=SEED
                ),
                tf.keras.layers.RandomContrast(
                    factor=0.05,  # ±5% contrast
                    seed=SEED
                )
            ])

            def apply_augmentation(image, label):
                if tf.random.uniform([], seed=SEED) < 0.5:  # 50% chance to apply augmentation
                    return trainAug(image), label
                return image, label

            train_ds = train_ds.map(apply_augmentation, 
                                   num_parallel_calls=NUM_PARALLEL_CALLS)

        # create batch and pre-fetch so one batch is always available
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(1)

    if args["backbone"] == "Xception":
        # Xception backbone
        model = build_with_Xception(input_shape=INPUT_SHAPE_RGB, output_nodes=NUM_CLASSES, refine=args["refine_Xception"])
        print("\nINFO: Using Xception backbone")
    elif args["backbone"] == "Efficient":
        # EfficientNetB7 backbone
        model = build_with_EfficientNet(input_shape=INPUT_SHAPE_RGB, output_nodes=NUM_CLASSES)
        print("\nINFO: Using EfficientNet backbone")
    else:
        # Simple CNN
        model = build_simple_cnn(input_shape=INPUT_SHAPE_RGB, output_nodes=NUM_CLASSES)
        print("\nINFO: Using simple CNN architecture")

    if args["checkpoint"]:
        print(f"\nINFO: Loading weights from checkpoint: {args['checkpoint']}")
        try:
            model.load_weights(args["checkpoint"]).expect_partial()
            print("INFO: Successfully loaded weights from checkpoint")
            
            if args["refine_Xception"] and args["backbone"] == "Xception":
                print("INFO: Unfreezing Xception backbone for fine-tuning")
                # Find the Xception base model by name instead of type
                base_model = None
                for layer in model.layers:
                    if 'xception' in layer.name.lower():
                        base_model = layer
                        break
                    
                if base_model is not None:
                    # Unfreeze the backbone
                    base_model.trainable = True
                    # Update the model's training phase
                    base_model._trainable = True  # Ensure the trainable flag is set at the layer level
                    print("INFO: Successfully unfroze Xception backbone")
                else:
                    print("WARNING: Could not find Xception backbone layer")
                    
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            print("INFO: Full checkpoint path:", os.path.abspath(args["checkpoint"]))
            print("INFO: Directory contents:", os.listdir(os.path.dirname(args["checkpoint"])))
            exit(1)

    model.compile(loss=LOSS,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy", MAPE])  # [LOSS, "mean_squared_error", "mean_absolute_error"])
    model.summary()

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = args["output_dir"] + "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    callbacks = [
        # Write TensorBoard logs to './logs' directory
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        # Write out trained weights every 5 epochs. This is apparently the "right" way to do this,
        # as the use of "save_weights='epoch' and "period=5" is deprecated smh...
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                    verbose=1,
        #                                    save_weights_only=True,
        #                                    save_freq=int(np.floor((len(X_train)/BATCH_SIZE))*SAVE_WEIGHTS_EVERY))
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           verbose=VERBOSE,
                                           save_weights_only=True,
                                           save_freq="epoch",
                                           period=SAVE_WEIGHTS_EVERY)
    ]

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # fit the model
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        verbose=VERBOSE,
                        callbacks=callbacks)

    print("\n------------------------------------------------------------------------")
    print("\nINFO: Completed training! Saved models:\n\n", os.listdir(checkpoint_dir))

    with open(checkpoint_dir + '/trainHistoryDict.pkl', 'wb') as train_hist:
        pickle.dump(history.history, train_hist)

    """
    get final test scores (MAPE_class and MAPE_true)
    """

    print("\nINFO: Evaluating trained model...\n")
    # load paths
    if TEST_DATA:
        X_test, y_test, y_gt_test, NUM_CLASSES = importLabeledImages(TEST_DATA,
                                                                     make_five_class=FIVE_CLASS)
        print("\nINFO: Reporting Final stats on supplied TEST dataset:", TEST_DATA)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    else:
        print("\nWARNING: NO TEST DATA SUPPLIED! REPORTING FINAL STATS ON TRAINING DATASET!")
        X_paths, y_paths, y_gt, NUM_CLASSES = importLabeledImages(DATA_PATH)
        test_ds = tf.data.Dataset.from_tensor_slices((X_paths, y_paths))
        X_test, y_test, y_gt_test = X_paths, y_paths, y_gt

    test_ds = test_ds.map(parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)
    test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)),
                          num_parallel_calls=NUM_PARALLEL_CALLS)
    # create batch and pre-fetch so one batch is always available
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(1)

    score = model.evaluate(test_ds, verbose=VERBOSE)
    y_pred = model.predict(test_ds, verbose=VERBOSE)
    print("\nINFO: Computing MAPE true...\n")
    MAPE_score_true, MAPE_score_ideal = MAPE_true(y_gt_test, y_pred)

    # get predicted class from one_hot encoded prediction vector
    y_pred_class = np.argmax(y_pred, axis=1)

    if FIVE_CLASS:
        class_list = [0.0013, 0.0030, 0.0068, 0.0154, 0.0351]
    else:
        class_list = [0.0010, 0.0012, 0.0015, 0.0019, 0.0023,
                      0.0028, 0.0034, 0.0042, 0.0052, 0.0064,
                      0.0078, 0.0096, 0.0118, 0.0145, 0.0179,
                      0.0219, 0.0270, 0.0331, 0.0407, 0.0500]

    y_pred_class_val = np.take(class_list, np.argmax(y_pred, axis=1))

    class_list_str = ["class_activation_" + str(i) for i in class_list]

    print("\nINFO: Final classification accuracy: %.4f" % score[1])
    print("INFO: Dataset IDEAL MAPE: %.2f" % MAPE_score_ideal)
    print("INFO: Final CLASS MAPE:   %.2f" % score[2])
    print("INFO: Final TRUE  MAPE:   %.2f" % MAPE_score_true)

    print("\n------------------------------------------------------------------------")

    print("\nINFO: Exporting results...")

    # save out all final results in separate file, just to be save.
    with open(args["output_dir"] + '/scores.txt', 'w') as f:
        with redirect_stdout(f):
            print("INFO: Training Settings\n",
                  "\nEPOCHS: ", EPOCHS,
                  "\nSAVE_WEIGHTS_EVERY: ", SAVE_WEIGHTS_EVERY,
                  "\nBATCH_SIZE: ", BATCH_SIZE,
                  "\nVERBOSE: ", VERBOSE,
                  "\nOPTIMIZER: ", OPTIMIZER,
                  "\nLEARNING_RATE: ", LEARNING_RATE,
                  "\nLOSS: ", LOSS,
                  "\nDATA_PATH: ", DATA_PATH,
                  "\nTEST_DATA: ", TEST_DATA,
                  "\nAUGMENTATION: ", AUGMENTATION,
                  "\nSEED: ", SEED)
            
            print("INFO: Final classification accuracy: %.4f" % score[1])
            print("INFO: Dataset IDEAL MAPE: %.2f" % MAPE_score_ideal)
            print("INFO: Final CLASS MAPE:   %.2f" % score[2])
            print("INFO: Final TRUE  MAPE:   %.2f" % MAPE_score_true)

    out_df = pd.DataFrame({"file": X_test,
                           "gt_class": y_test,
                           "pred_class": y_pred_class,
                           "gt": y_gt_test,
                           "pred": y_pred_class_val})

    # add activations to output file to show confidence distribution
    for cl, class_entry in enumerate(class_list_str):
        out_df[class_entry] = y_pred[:, cl]

    out_df.to_csv(args["output_dir"] + '/test_data_pred_results.csv')

    print("INFO: Export complete.")
