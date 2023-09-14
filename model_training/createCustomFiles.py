import math
import numpy as np
import os
import csv
from imutils import paths
from sklearn.utils import shuffle
import argparse


def createCustomFiles(obIDs=["Ant"], amountTest=0.1, random_state=0, k_fold=[5, None], output_folder="",
                      custom_name=None, shuffle_files=False, backup_location="backup"):
    """
    Creates custom folder and files for training and testing YOLOv3 & YOLOv4 with Darknet Framework
    :param obIDs:List of names of objects
    :param amountTest: amount of testing data to be withheld from training
    :param output_folder: specify output folder if desired
    """

    # depending on the specified amountTest, assume different output folder structures
    if amountTest == 1:
        file_dir = "obj_test"
    else:
        file_dir = "obj"

    if custom_name is not None:
        paths_folders = [output_folder + "data",
                         output_folder + "data/" + file_dir,
                         output_folder + "data"] * 3
        train_file = "data/" + custom_name + "_train.txt"
        test_file = "data/" + custom_name + "_test.txt"
        names_file = "data/obj.names"
    else:
        paths_folders = [output_folder + "data",
                         output_folder + "data/" + file_dir,
                         output_folder + "data"] * 3
        train_file = "data/train.txt"
        test_file = "data/test.txt"
        names_file = "data/obj.names"

    if len(obIDs) != 1:
        print("Using custom labels:", obIDs)

    # create all required folders
    for folder in range(len(paths_folders)):
        if not os.path.exists(paths_folders[folder]):
            os.mkdir(paths_folders[folder])

    # create object file (contains list of names, corresponding to objectIDs)
    with open(paths_folders[2] + "/" + "obj.names", "w") as f:
        for ob in range(len(obIDs)):
            f.write(str(obIDs[ob]) + "\n")

    # create trainer.data fle based on inputs
    if custom_name is not None:
        with open(paths_folders[2] + "/" + custom_name, "w") as f:
            f.write("classes = " + str(len(obIDs)) + "\n")
            f.write("train = " + train_file + "\n")
            f.write("test = " + test_file + "\n")
            f.write("names = " + names_file + "\n")
            f.write("backup = " + backup_location + "/\n")
    else:
        with open(paths_folders[2] + "/" + "obj.data", "w") as f:
            f.write("classes = " + str(len(obIDs)) + "\n")
            f.write("train = " + train_file + "\n")
            f.write("test = " + test_file + "\n")
            f.write("names = " + names_file + "\n")
            f.write("backup = " + backup_location + "/\n")

    files = []
    labels = []

    # r=root, d=directories, f = files

    for r, d, f in os.walk(str(paths_folders[1] + "/")):
        for file in f:
            if '.txt' in file:
                labels.append(os.path.join(r, file))

    for imagePath in sorted(paths.list_images(paths_folders[1])):
        files.append(imagePath)

    # set a fixed seed, so results can be replicated by enforcing the same splits every time the script is executed
    # the optional parameter 'random_state' can be used to set a fixed seed. (By default "np.random")
    if k_fold[1] is None:
        if shuffle_files:
            files, labels = shuffle(files, labels, random_state=random_state)
        num_train_examples = int(np.floor(len(files) * (1 - amountTest)))

        print("Using", num_train_examples, "training images and",
              int(np.floor(len(files) - (len(files) * (1 - amountTest)))), "test images. (" + str(amountTest * 100),
              "%)")

        files_train, labels_train = files[0:num_train_examples], labels[0:num_train_examples]
        files_test, labels_test = files[num_train_examples:], labels[num_train_examples:]
    else:
        if len(files) > k_fold[0]:
            files, labels = shuffle(files, labels, random_state=0)
            # use k_fold crossvalidation and return the defined split for Test and Train data
            kf = KFold(n_splits=k_fold[0])
            # return the defined split
            kf_id = 0
            for train_index, test_index in kf.split(labels):
                if kf_id == k_fold[1]:
                    files_train = [files[i] for i in train_index]
                    labels_train = [labels[i] for i in train_index]
                    files_test = [files[i] for i in test_index]
                    labels_test = [labels[i] for i in test_index]
                    break
                else:
                    kf_id += 1

            print("Using", len(train_index), "training images and",
                  len(test_index), "test images. (", round(100 / k_fold[0]), "%)")
        else:
            # if fewer files are passed then required for the defined split, return empty lists
            files_train, files_test = [], []

    # create train.txt and test.txt files, containing the locations of the respective image files
    if custom_name is not None:
        if len(files_train) > 0:
            with open(paths_folders[2] + "/" + custom_name + "_train.txt", "w") as f:
                for file in range(len(files_train)):
                    f.write("data/" + file_dir + "/" + files_train[file].split(SEPARATOR)[-1] + "\n")
        if len(files_test) > 0:
            with open(paths_folders[2] + "/" + custom_name + "_test.txt", "w") as f:
                for file in range(len(files_test)):
                    f.write("data/" + file_dir + "/" + files_test[file].split(SEPARATOR)[-1] + "\n")
    else:
        if len(files_train) > 0:
            with open(paths_folders[2] + "/" + "train.txt", "w") as f:
                for file in range(len(files_train)):
                    f.write("data/" + file_dir + "/" + files_train[file].split(SEPARATOR)[-1] + "\n")
        if len(files_test) > 0:
            with open(paths_folders[2] + "/" + "test.txt", "w") as f:
                for file in range(len(files_test)):
                    f.write("data/" + file_dir + "/" + files_test[file].split(SEPARATOR)[-1] + "\n")

    print("Successfully created all required files!")


if __name__ == "__main__":
    # this script assumes that YOLO conform annotated training and testing images have already
    # been created anb only the folder strutcure train.txt and test.txt files need to be created
    # in this particular implementation, the training and testing samples are saved in separate
    # folders
    if os.name == 'posix':
        SEPARATOR = "/"
    else:
        SEPARATOR = "\\"

    classes = [0.0010, 0.0012, 0.0015, 0.0019, 0.0023, 0.0028,
               0.0034, 0.0042, 0.0052, 0.0064, 0.0078, 0.0096,
               0.0118, 0.0145, 0.0179, 0.0219, 0.0270, 0.0331,
               0.0407, 0.0500]

    classes_str = [str(i) for i in classes]

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-t", "--amount_test", default=0.1, type=float)
    ap.add_argument("-k", "--k_fold", default=5, type=int)
    ap.add_argument("-r", "--rand_seed", default=0, required=False, type=int)
    ap.add_argument("-o", "--output_dir", required=True, type=str)
    ap.add_argument("-s", "--shuffle", default=False, required=False, type=bool)
    ap.add_argument("-b", "--backup_location", default="backup", required=False, type=str)

    args = vars(ap.parse_args())

    # reference execution from YOLO_export
    createCustomFiles(obIDs=classes_str,
                      amountTest=float(args["amount_test"]),
                      random_state=int(args["rand_seed"]),
                      k_fold=[args["k_fold"], None],
                      output_folder=args["output_dir"] + "/",
                      shuffle_files=args["shuffle"],
                      backup_location=args["backup_location"])
