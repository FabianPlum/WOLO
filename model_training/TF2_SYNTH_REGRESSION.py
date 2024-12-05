import os  # to perform file operations
from os.path import isfile, join
import argparse
import pickle
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
def importLabeledImages(target_directory):
    X = []
    y = []
    for cl, folder in enumerate(os.listdir(target_directory)):
        print(join(target_directory, folder))
        for filename in os.listdir(join(target_directory, folder)):
            X.append(join(target_directory, folder, filename))

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
            y.append(y_val)

    return X, y


def MAPE(y_true, y_pred, scaling_factor=1, min_val=0):
    """
    Compute MAPE with protections for near-zero values and optional scaling
    
    Designed to work with:
    1. Sigmoid-scaled outputs (0-1 range)
    2. Log-transformed data (when LOG=True)
    3. Near-zero target values
    
    Parameters:
        y_true: Ground truth values (batch x output_dims)
        y_pred: Predicted values (batch x output_dims)
        scaling_factor: Multiplier to denormalize values
        min_val: Offset for denormalization
    
    Returns:
        tf.Tensor: MAPE score (percentage error)
    
    Note:
        - Uses floor value of 1e-5 to prevent division by very small numbers
        - Automatically handles log-transformed data if LOG=True
        - Applies denormalization before computing error if scaling_factor != 1
    """
    if LOG:
        y_true = delog_and_denorm(y_true)
        y_pred = delog_and_denorm(y_pred)

    # Add floor value to prevent division by very small numbers
    floor = tf.maximum((y_true + min_val) * scaling_factor, 1e-5)
    
    APE = tf.math.abs(
        tf.math.divide(
            ((y_true + min_val) * scaling_factor - (y_pred + min_val) * scaling_factor),
            floor
        )
    )
    MAPE = tf.math.multiply(tf.constant(100, tf.float32), tf.math.reduce_mean(APE, axis=-1))
    return MAPE


def log_and_norm(y, y_min=0.0001, y_max=0.05, y_range=[0, 1]):
    y_log = tf.math.log(y)
    y_min_log = tf.math.log(y_min)
    y_max_log = tf.math.log(y_max)

    y_normalised = y_range[0] + ((y_range[1] - y_range[0]) / (y_max_log - y_min_log)) * (y_log - y_min_log)

    return y_normalised


def delog_and_denorm(y_ln, y_min=0.0001, y_max=0.05, y_range=[0, 1]):
    y_min_log = tf.math.log(y_min)
    y_max_log = tf.math.log(y_max)

    y_log = (y_ln - y_range[0]) / ((y_range[1] - y_range[0]) / (y_max_log - y_min_log)) + y_min_log

    y = tf.math.exp(y_log)

    return y


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


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

    # Raw dense output followed by scaling
    dense_out = keras.layers.Dense(output_nodes, name='dense_out')(x3_d)
    outputs = keras.layers.Lambda(lambda x: tf.nn.sigmoid(x), name='output')(dense_out)

    model = keras.Model(inputs, outputs)

    return model


def ALT_build_with_EfficientNet(input_shape, output_nodes=1, refine=False):
    # Load weights pre-trained on ImageNet.
    # Do not include the ImageNet classifier at the top.
    base_model = keras.applications.efficientnet.EfficientNetB7(weights="imagenet",
                                                                input_shape=input_shape,
                                                                include_top=False,
                                                                pooling="avg")

    # freeze the weights of the backbone
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
    xa = base_model(x, training=False)
    # x1 = keras.layers.GlobalAveragePooling2D()(xa)
    x1_d = tf.keras.layers.Dropout(.2)(xa)
    # A Dense classifier with a single unit
    x2 = keras.layers.Dense(4096, activation="relu")(x1_d)  # 1024
    x2_d = tf.keras.layers.Dropout(.2)(x2)

    x3 = keras.layers.Dense(4096, activation="relu")(x2_d)  # 1024
    x3_d = tf.keras.layers.Dropout(.2)(x3)

    outputs = keras.layers.Dense(output_nodes)(x3_d)

    model = keras.Model(inputs, outputs)

    return model


def build_with_EfficientNet(input_shape, output_nodes=1, refine=False):
    inputs = keras.Input(shape=input_shape)

    base_model = keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                                input_shape=input_shape,
                                                                input_tensor=inputs,
                                                                weights="imagenet")

    # Freeze the pretrained weights
    base_model.trainable = refine

    # Rebuild top
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    # Raw dense output followed by scaling
    dense_out = keras.layers.Dense(output_nodes, name='dense_out')(x)
    outputs = keras.layers.Lambda(lambda x: tf.nn.sigmoid(x), name='output')(dense_out)

    model = keras.Model(inputs, outputs)

    return model


def build_simple_cnn(input_shape, output_nodes=1):
    """
    Build a simple CNN for regression with scaled outputs
    
    Architecture includes:
    - Standard CNN backbone with regularization
    - Final dense layer without activation
    - Sigmoid scaling layer to constrain outputs to (0,1)
    
    Parameters:
        input_shape (tuple): Shape of input images (height, width, channels)
        output_nodes (int): Number of output values to predict
    
    Returns:
        keras.Model: Compiled model with sigmoid-scaled outputs
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
    
    # Output layers with controlled scaling
    dense_out = keras.layers.Dense(output_nodes, name='dense_out')(drop_dense1)  # Raw output
    outputs = keras.layers.Lambda(lambda x: tf.nn.sigmoid(x), name='output')(dense_out)  # Scale to (0,1)
    
    model = keras.Model(inputs, outputs, name='simple_cnn')
    return model


def custom_regression_loss(alpha=0.7, beta=0.3, epsilon=1e-5):
    """
    Custom loss combining MAPE and MSE components for regression with near-zero values
    
    The loss function is designed to handle regression tasks where:
    1. Target values can be very close to zero
    2. Both relative (MAPE) and absolute (MSE) errors are important
    3. Training stability is crucial
    
    Parameters:
        alpha (float): Weight for MAPE component (0-1)
        beta (float): Weight for MSE component (0-1)
        epsilon (float): Minimum floor value to prevent division by very small numbers
    
    Returns:
        function: Loss function that takes (y_true, y_pred) and returns combined loss
    
    Note:
        - alpha + beta should equal 1.0 for proper weighting
        - epsilon should be set based on the minimum meaningful value in your dataset
    """
    def loss(y_true, y_pred):
        # Floor value prevents division by very small numbers
        floor = tf.maximum(y_true, epsilon)
        
        # MAPE component with protection against near-zero division
        mape = tf.reduce_mean(tf.abs((y_true - y_pred) / floor)) * 100
        
        # MSE component provides stability and handles absolute errors
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Weighted combination of both components
        total_loss = alpha * mape + beta * mse
        
        return total_loss
    
    return loss


if __name__ == "__main__":
    # limit number of simultaneously used cores / threads
    # os.environ["OPENBLAS_NUM_THREADS"] = "8"  # export OPENBLAS_NUM_THREADS=8

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-d", "--dataset", required=True, type=str)
    ap.add_argument("-r", "--rand_seed", default=0, required=False, type=int)
    ap.add_argument("-o", "--output_dir", required=True, type=str)
    ap.add_argument("-b", "--backbone", default=None, required=False, type=str)
    ap.add_argument("-l", "--LOSS", default="MAPE", required=False, type=str)

    # Optional flags
    ap.add_argument("-e", "--epochs", default=10, required=False, type=int)
    ap.add_argument("-bs", "--batch_size", default=128, required=False, type=int)
    ap.add_argument("-sw", "--save_weights_every", default=10, required=False, type=int)
    ap.add_argument("-aug", "--augmentation", default=False, required=False, type=bool)
    ap.add_argument("-log", "--log_transform", default=False, required=False, type=bool)
    ap.add_argument("-t", "--test", default=False, required=False, type=str)
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
    OPTIMIZER = "adam"
    LOSS = args["LOSS"]
    LOG = args["log_transform"]
    IMG_ROWS, IMG_COLS = 128, 128
    INPUT_SHAPE_RGB = (IMG_ROWS, IMG_COLS, 3)
    NUM_PARALLEL_CALLS = tf.data.AUTOTUNE  # -> use AUTOTUNE only on GPU, otherwise it wants ALL the CPUs on HPC
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
          "\nBACKBONE: ", args["backbone"],
          "\nOPTIMIZER: ", OPTIMIZER,
          "\nLEARNING_RATE: ", LEARNING_RATE,
          "\nLOSS: ", LOSS,
          "\nLOG_DATA", LOG,
          "\nDATA_PATH: ", DATA_PATH,
          "\nTEST_DATA: ", TEST_DATA,
          "\nAUGMENTATION: ", AUGMENTATION,
          "\nSEED: ", SEED)

    if args["augmentation"] == True:
        print("\nINFO: Data augmentation - enabled\n")
    else:
        print("\nINFO: Data augmentation - disabled\n")

    print("\n--------------------------------------\n")

    # set all randomisation seeds for reproducible results
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    # use CPU for all pre-processing
    with tf.device('/cpu:0'):
        # load paths
        X_paths, y_paths = importLabeledImages(DATA_PATH)
        NUM_CLASSES = 1

        # 1 shuffle
        X_paths, y_paths = shuffle(X_paths, y_paths, random_state=SEED)

        # 1.5 balance classes

        train_ds = tf.data.Dataset.from_tensor_slices((X_paths, y_paths))
        # normally shuffle should be equal to len(filenames), but as we shuffle in the preprocessing step
        # BATCH_SIZE * 100 is acceptable here
        train_ds = train_ds.shuffle(BATCH_SIZE * 100)
        train_ds = train_ds.map(parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)

        # apply log transform, if enabled
        if args["log_transform"]:
            train_ds = train_ds.map(lambda x, y: (x, log_and_norm(y)),
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
        model = build_with_Xception(input_shape=INPUT_SHAPE_RGB, output_nodes=NUM_CLASSES)
        print("\nINFO: Using Xception backbone")
    elif args["backbone"] == "Efficient":
        # EfficientNetB7 backbone
        model = build_with_EfficientNet(input_shape=INPUT_SHAPE_RGB, output_nodes=NUM_CLASSES)
        print("\nINFO: Using EfficientNet backbone")
    else:
        # Simple CNN
        model = build_simple_cnn(input_shape=INPUT_SHAPE_RGB, output_nodes=NUM_CLASSES)
        print("\nINFO: Using simple CNN architecture")

    if LOSS.split("_")[0] == "custom":
        alpha = float(LOSS.split("_")[1])
        beta = float(LOSS.split("_")[2])
        LOSS = custom_regression_loss(alpha=alpha, beta=beta)

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

    model.compile(
        loss=LOSS,
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[MAPE]
    )
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
        X_test, y_test = importLabeledImages(TEST_DATA)
        print("\nINFO: Reporting Final stats on supplied TEST dataset:", TEST_DATA)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    else:
        print("\nWARNING: NO TEST DATA SUPPLIED! REPORTING FINAL STATS ON TRAINING DATASET!")
        X_paths, y_paths = importLabeledImages(DATA_PATH)
        test_ds = tf.data.Dataset.from_tensor_slices((X_paths, y_paths))
        X_test, y_test = X_paths, y_paths

    test_ds = test_ds.map(parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)
    # apply log transform, if enabled
    if args["log_transform"]:
        test_ds = test_ds.map(lambda x, y: (x, log_and_norm(y)),
                              num_parallel_calls=NUM_PARALLEL_CALLS)
    # create batch and pre-fetch so one batch is always available
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(1)

    score = model.evaluate(test_ds, verbose=VERBOSE)
    y_pred = model.predict(test_ds, verbose=VERBOSE)
    if LOG:
        y_pred = delog_and_denorm(y_pred)
        y_pred_out = y_pred.numpy().reshape(tf.shape(y_pred)[0])
    else:
        y_pred_out = y_pred.reshape(len(y_pred))

    print("INFO: Final TRUE MAPE:   %.2f" % score[1])

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
                  "\nLOG_DATA", LOG,
                  "\nDATA_PATH: ", DATA_PATH,
                  "\nTEST_DATA: ", TEST_DATA,
                  "\nAUGMENTATION: ", AUGMENTATION,
                  "\nSEED: ", SEED)

            print("\nINFO: Final TRUE MAPE:   %.2f" % score[1])

    out_df = pd.DataFrame({"file": X_test,
                           "gt": y_test,
                           "pred": y_pred_out})

    out_df.to_csv(args["output_dir"] + '/test_data_pred_results.csv')

    print("INFO: Export complete.")
