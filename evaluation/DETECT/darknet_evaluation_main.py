import numpy as np
import pickle
import os
from os import listdir
from os.path import join
from pathlib import Path
import subprocess
import argparse
import time

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-md", "--modelFolder", required=True, type=str)
    ap.add_argument("-dt", "--dataFolder", required=True, type=str)
    ap.add_argument("-of", "--outputFolder", required=False, type=str,
                    default=Path(__file__).parent.resolve())
    ap.add_argument("-da", "--darknetFolder", required=True, type=str)

    # Darknet setup
    ap.add_argument("-c", "--configPath", required=True, type=str)
    ap.add_argument("-m", "--metaPath", required=True, type=str)
    ap.add_argument("-min", "--min_size", default=0.1, required=False, type=int)
    ap.add_argument("-so", "--showDetections", default="", required=False, type=bool)
    ap.add_argument("-GPU", "--GPU", default="0", required=False, type=str)

    ap.add_argument("-l", "--lastOnly", default=False, required=False, type=bool)
    ap.add_argument("-cr", "--CROPPED", default=False, required=False, type=bool)

    args = vars(ap.parse_args())

    # set which GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["GPU"]

    # Data input and output
    modelFolder = args["modelFolder"]
    dataFolder = args["dataFolder"]
    outputFolder = args["outputFolder"]
    darknetFolder = args["darknetFolder"]

    # evaluate only the final trained state
    last_only = args["lastOnly"]

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    start_time = time.time()

    print("Writing to output folder:", outputFolder, "\n")

    # Darknet setup
    configPath = args["configPath"]
    metaPath = args["metaPath"]
    min_size = int(args["min_size"])
    showDetections = args["showDetections"]

    model_folder = Path(args["modelFolder"])
    # get all weights files
    if last_only:
        model_paths = [join(model_folder, f) for f in listdir(model_folder)
                       if
                       str(join(model_folder, f)).split(".")[-1] == "weights" and
                       str(join(model_folder, f)).split(".")[0][
                           -1] == "t"]
    else:
        model_paths = [join(model_folder, f) for f in listdir(model_folder)
                       if
                       str(join(model_folder, f)).split(".")[-1] == "weights" and
                       str(join(model_folder, f)).split(".")[0][
                           -1] == "0"]

        model_paths.sort()

    print("\nFound a total of {} trained models".format(len(model_paths)))

    sample_folder = Path(args["dataFolder"])
    if args["CROPPED"]:
        # get test sample files
        sample_paths = []
        for path, subdirs, files in os.walk(sample_folder):
            for name in files:
                sample_paths.append(os.path.join(path, name))

    else:
        # get test sample files
        sample_paths = [join(sample_folder, f) for f in listdir(sample_folder)
                        if str(join(sample_folder, f)).split(".")[-1] == "jpg"]

    print("\nFound a total of {} samples".format(len(sample_paths)))

    # read config file to determine network (input) shape
    network_shape = [None, None]
    f = open(configPath, 'r')
    Lines = f.readlines()
    # Strips the "\n" newline character
    for line in Lines:
        line_cleaned = line.strip()
        line_arr = line_cleaned.split("=")
        if line_arr[0] == "width":
            network_shape[0] = int(line_arr[1])
        if line_arr[0] == "height":
            network_shape[1] = int(line_arr[1])

        if network_shape[0] is not None and network_shape[1] is not None:
            break

    f.close()
    print("Network shape:", network_shape)

    # CALLING SUB_DARKNET_WOLO.py TO PERFORM ALL DETECTIONS

    all_detections = []

    for weightPath in model_paths:
        if args["CROPPED"]:
            output_name = str(os.path.join(outputFolder,
                                           os.path.basename(os.path.dirname(model_folder))
                                           + "_" + str(os.path.basename(weightPath)).split(".")[0]
                                           + "_" + str(os.path.basename(sample_folder)) + "_128"))
        else:
            output_name = str(os.path.join(outputFolder,
                                           os.path.basename(os.path.dirname(model_folder))
                                           + "_" + str(os.path.basename(weightPath)).split(".")[0]))
        print(output_name)

        if showDetections:
            if args["CROPPED"]:
                subprocess.call(['python', 'sub_darknet_WOLO.py',
                                 "--darknetFolder", darknetFolder,
                                 "--configPath", configPath,
                                 "--weightPath", weightPath,
                                 "--metaPath", metaPath,
                                 "--samplePath", str(sample_folder),
                                 "--outputName", output_name,
                                 "--min_size", str(10),
                                 "--showDetections", "True",
                                 "--includeSample", "True",
                                 "--CROPPED", "True"])
            else:
                subprocess.call(['python', 'sub_darknet_WOLO.py',
                                 "--darknetFolder", darknetFolder,
                                 "--configPath", configPath,
                                 "--weightPath", weightPath,
                                 "--metaPath", metaPath,
                                 "--samplePath", str(sample_folder),
                                 "--outputName", output_name,
                                 "--min_size", str(10),
                                 "--showDetections", "True",
                                 "--includeSample", "True"])
        else:
            if args["CROPPED"]:
                subprocess.call(['python', 'sub_darknet_WOLO.py',
                                 "--darknetFolder", darknetFolder,
                                 "--configPath", configPath,
                                 "--weightPath", weightPath,
                                 "--metaPath", metaPath,
                                 "--samplePath", str(sample_folder),
                                 "--outputName", output_name,
                                 "--min_size", str(10),
                                 "--includeSample", "True",
                                 "--CROPPED", "True"])
            else:
                subprocess.call(['python', 'sub_darknet_WOLO.py',
                                 "--darknetFolder", darknetFolder,
                                 "--configPath", configPath,
                                 "--weightPath", weightPath,
                                 "--metaPath", metaPath,
                                 "--samplePath", str(sample_folder),
                                 "--outputName", output_name,
                                 "--min_size", str(10),
                                 "--includeSample", "True"])

        with open(os.path.join(outputFolder, output_name + ".pkl"), 'rb') as f:
            all_detections.append([output_name, pickle.load(f)])

        print("ran inference on {} frames, using {}".format(len(all_detections[-1][1]), output_name))

    print("--- %s seconds ---" % (time.time() - start_time))
