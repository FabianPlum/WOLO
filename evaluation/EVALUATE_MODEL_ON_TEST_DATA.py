"""

# basic structure:
# read out model parameters from model name (inference type, loss function, etc)
# create model function according to read info
model = create_model()

# Loads the weights
model.load_weights(checkpoint_path)

# evaluate the model / predict labels
loss, acc = model.evaluate(test_images, test_labels, verbose=2)

# compose new name from model input + test data name
# export predictions and scores to new folder

"""

import argparse
import os
import tensorflow as tf

def get_latest_checkpoint(model_path):
    """Find the latest checkpoint file in the given directory.
    
    Args:
        model_path: Path to directory containing checkpoint files
        
    Returns:
        String path to latest checkpoint without the .index extension
    """
    checkpoints = [f for f in os.listdir(model_path) if f.endswith('.ckpt.index')]
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1].split('.')[0]))
    print("\n------------------------------------------------------\n",
          "INFO: Latest checkpoint:", latest_checkpoint,
          "\n------------------------------------------------------\n")
    return latest_checkpoint[:-6]  # Remove .index suffix

import sys
sys.path.append("../model_training")
sys.path.append("model_training")

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-i", "--input_model", required=True, type=str)
    ap.add_argument("-d", "--dataset", required=True, type=str)
    ap.add_argument("-o", "--output_dir", required=True, type=str)

    args = vars(ap.parse_args())

    BATCH_SIZE = 256 #adjust this when running out of memory (tests ran on RTX 4090)
    VERBOSE = 2
    LOSS = "categorical_crossentropy"
    OPTIMIZER = "adam"
    NUM_PARALLEL_CALLS = tf.data.AUTOTUNE
    MODEL_PATH = args["input_model"]
    MODEL_NAME = str(os.path.basename(MODEL_PATH))
    IMG_ROWS, IMG_COLS = 128, 128
    INPUT_SHAPE_RGB = (IMG_ROWS, IMG_COLS, 3)
    DATA_PATH = args["dataset"]
    OUTPUT_DIR = os.path.join(args["output_dir"],
                              MODEL_NAME + "---" +
                              str(os.path.basename(DATA_PATH)))

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if MODEL_NAME.split("_")[0] == "CLASS":
        if MODEL_NAME.split("_")[2] == "20":
            INFERENCE_METHOD = "CLASS_20"
            from TF2_SYNTH_CLASSIFICATION import *
        elif MODEL_NAME.split("_")[2] == "5":
            INFERENCE_METHOD = "CLASS_5"
            from TF2_SYNTH_CLASSIFICATION_5_CLASS import *
            import TF2_SYNTH_CLASSIFICATION_5_CLASS

        if MODEL_NAME.split("_")[5] == "Vanilla":
            INFERENCE_METHOD = INFERENCE_METHOD + "_VANILLA"

    else:
        INFERENCE_METHOD = "REG"
        from TF2_SYNTH_REGRESSION import *
        import TF2_SYNTH_REGRESSION

        if MODEL_NAME.split("_")[-3] == "Vanilla":
            INFERENCE_METHOD = INFERENCE_METHOD + "_VANILLA"

    print("INFO: Evaluating", MODEL_NAME, "on", str(os.path.basename(DATA_PATH)))
    print("INFO: Inference method:", INFERENCE_METHOD)

    if INFERENCE_METHOD[:3] == "REG":
        if "COMB" in MODEL_NAME:
            # this is somewhat hard-coded but it's the only combination tested here.
            LOSS = custom_regression_loss(alpha=0.5, beta=0.5)
        else:
            LOSS = MODEL_NAME.split("_")[2]

        if "LOG" in MODEL_NAME:
            LOG = True
            TF2_SYNTH_REGRESSION.LOG = True
        else:
            LOG = False
            TF2_SYNTH_REGRESSION.LOG = False

        print("\nINFO: Loading model...\n")
        if INFERENCE_METHOD == "REG_VANILLA":
            print("INFO: Using simple CNN architecture")
            model = build_simple_cnn(input_shape=INPUT_SHAPE_RGB, output_nodes=1)
        else:
            print("INFO: Using Xception backbone")
            model = build_with_Xception(input_shape=INPUT_SHAPE_RGB, output_nodes=1)
            
        model.load_weights(os.path.join(MODEL_PATH, get_latest_checkpoint(MODEL_PATH))).expect_partial()

        model.compile(loss=LOSS,
                      optimizer=OPTIMIZER,
                      metrics=[MAPE])  # [LOSS, "mean_squared_error", "mean_absolute_error"])
        model.summary()

        print("\nINFO: Evaluating trained model...\n")
        # load paths
        X_test, y_test = importLabeledImages(DATA_PATH)
        print("\nINFO: Reporting Final stats on supplied TEST dataset:", DATA_PATH)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        test_ds = test_ds.map(parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)
        # apply log transform, if enabled
        if LOG:
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

        # save out all final results in separate file, just to be safe.
        with open(os.path.join(OUTPUT_DIR, "scores.txt"), 'w') as f:
            with redirect_stdout(f):
                print("INFO: Training Settings\n",
                      "\nOPTIMIZER: ", OPTIMIZER,
                      "\nLOSS: ", LOSS,
                      "\nLOG_DATA", LOG,
                      "\nTRAINING_PATH: ", MODEL_NAME.split("_")[1],
                      "\nDATA_PATH: ", DATA_PATH)

                print("\nINFO: Final TRUE MAPE:   %.2f" % score[1])

        out_df = pd.DataFrame({"file": X_test,
                               "gt": y_test,
                               "pred": y_pred_out})

        out_df.to_csv(os.path.join(OUTPUT_DIR, 'test_data_pred_results.csv'))

        print("INFO: Export complete.")

    if INFERENCE_METHOD == "CLASS_20" or INFERENCE_METHOD == "CLASS_20_VANILLA":
        print("\nINFO: Loading model...\n")
        if INFERENCE_METHOD == "CLASS_20":
            model = build_with_Xception(input_shape=INPUT_SHAPE_RGB, output_nodes=20)
            print("\nINFO: Using Xception backbone")
        else:
            model = build_simple_cnn(input_shape=INPUT_SHAPE_RGB, output_nodes=20)
            print("\nINFO: Using simple CNN architecture")

        model.load_weights(os.path.join(MODEL_PATH, get_latest_checkpoint(MODEL_PATH))).expect_partial()

        model.compile(loss=LOSS,
                      optimizer=OPTIMIZER,
                      metrics=["accuracy", MAPE])  # [LOSS, "mean_squared_error", "mean_absolute_error"])
        model.summary()

        print("\nINFO: Evaluating trained model...\n")
        # load paths
        X_test, y_test, y_gt_test, NUM_CLASSES = importLabeledImages(DATA_PATH)
        print("\nINFO: Reporting Final stats on supplied TEST dataset:", DATA_PATH)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        test_ds = test_ds.map(parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)
        test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)),
                              num_parallel_calls=NUM_PARALLEL_CALLS)
        # create batch and pre-fetch so one batch is always available
        test_ds = test_ds.batch(BATCH_SIZE)
        test_ds = test_ds.prefetch(1)

        score = model.evaluate(test_ds, verbose=VERBOSE)
        y_pred = model.predict(test_ds, verbose=VERBOSE)

        # get predicted class from one_hot encoded prediction vector
        y_pred_class = np.argmax(y_pred, axis=1)

        class_list = [0.0010, 0.0012, 0.0015, 0.0019, 0.0023,
                      0.0028, 0.0034, 0.0042, 0.0052, 0.0064,
                      0.0078, 0.0096, 0.0118, 0.0145, 0.0179,
                      0.0219, 0.0270, 0.0331, 0.0407, 0.0500]

        y_pred_class_val = np.take(class_list, np.argmax(y_pred, axis=1))

        class_list_str = ["class_activation_" + str(i) for i in class_list]

        print("\nINFO: Final classification accuracy: %.4f" % score[1])
        print("INFO: Final CLASS MAPE:   %.2f" % score[2])

        print("\n------------------------------------------------------------------------")

        print("\nINFO: Exporting results...")

        # save out all final results in separate file, just to be safe.
        with open(os.path.join(OUTPUT_DIR, 'scores.txt'), 'w') as f:
            with redirect_stdout(f):
                print("INFO: Final classification accuracy: %.4f" % score[1])
                print("INFO: Final CLASS MAPE:   %.2f" % score[2])

        out_df = pd.DataFrame({"file": X_test,
                               "gt_class": y_test,
                               "pred_class": y_pred_class,
                               "gt": y_gt_test,
                               "pred": y_pred_class_val})

        # add activations to output file to show confidence distribution
        for cl, class_entry in enumerate(class_list_str):
            out_df[class_entry] = y_pred[:, cl]

        out_df.to_csv(os.path.join(OUTPUT_DIR, 'test_data_pred_results.csv'))

        print("INFO: Export complete.")

    if INFERENCE_METHOD[:7] == "CLASS_5":
        print("\nINFO: Loading model...\n")
        if INFERENCE_METHOD == "CLASS_5":
            model = build_with_Xception(input_shape=INPUT_SHAPE_RGB, output_nodes=5)
        else:
            model = build_simple_cnn(input_shape=INPUT_SHAPE_RGB, output_nodes=5)

        model.load_weights(os.path.join(MODEL_PATH, get_latest_checkpoint(MODEL_PATH))).expect_partial()

        model.compile(loss=LOSS,
                      optimizer=OPTIMIZER,
                      metrics=["accuracy", MAPE])  # [LOSS, "mean_squared_error", "mean_absolute_error"])
        model.summary()

        TF2_SYNTH_CLASSIFICATION_5_CLASS.FIVE_CLASS = True

        print("\nINFO: Evaluating trained model...\n")
        # load paths
        X_test, y_test, y_gt_test, NUM_CLASSES = importLabeledImages(DATA_PATH)
        print("\nINFO: Reporting Final stats on supplied TEST dataset:", DATA_PATH)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        test_ds = test_ds.map(parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)
        test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)),
                              num_parallel_calls=NUM_PARALLEL_CALLS)
        # create batch and pre-fetch so one batch is always available
        test_ds = test_ds.batch(BATCH_SIZE)
        test_ds = test_ds.prefetch(1)

        score = model.evaluate(test_ds, verbose=VERBOSE)
        y_pred = model.predict(test_ds, verbose=VERBOSE)

        # get predicted class from one_hot encoded prediction vector
        y_pred_class = np.argmax(y_pred, axis=1)

        class_list = [0.0013, 0.0030, 0.0068, 0.0154, 0.0351]

        y_pred_class_val = np.take(class_list, np.argmax(y_pred, axis=1))

        class_list_str = ["class_activation_" + str(i) for i in class_list]

        print("\nINFO: Final classification accuracy: %.4f" % score[1])
        print("INFO: Final CLASS MAPE:   %.2f" % score[2])

        print("\n------------------------------------------------------------------------")

        print("\nINFO: Exporting results...")

        # save out all final results in separate file, just to be safe.
        with open(os.path.join(OUTPUT_DIR, 'scores.txt'), 'w') as f:
            with redirect_stdout(f):
                print("INFO: Final classification accuracy: %.4f" % score[1])
                print("INFO: Final CLASS MAPE:   %.2f" % score[2])

        out_df = pd.DataFrame({"file": X_test,
                               "gt_class": y_test,
                               "pred_class": y_pred_class,
                               "gt": y_gt_test,
                               "pred": y_pred_class_val})

        # add activations to output file to show confidence distribution
        for cl, class_entry in enumerate(class_list_str):
            out_df[class_entry] = y_pred[:, cl]

        out_df.to_csv(os.path.join(OUTPUT_DIR, 'test_data_pred_results.csv'))

        print("INFO: Export complete.")
