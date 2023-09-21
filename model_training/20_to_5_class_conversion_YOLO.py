import os
import argparse


def translate_line(line):
    lookup_table = [0, 0, 0, 0,
                    1, 1, 1, 1,
                    2, 2, 2, 2,
                    3, 3, 3, 3,
                    4, 4, 4, 4]

    try:
        line_elems = line.split(" ")
        new_class = lookup_table[int(line_elems[0])]

        new_line = str(new_class) + " " + " ".join(line_elems[1:])
    except:
        new_line = ""
    print(new_line)
    return new_line


def process_file(path):
    # open file in read mode
    print("\nINFO: updating file:", path)
    file = open(path, "r")
    replaced_content = ""
    # looping through the file
    for line in file:
        # stripping line break
        line = line.strip()
        # replacing the texts
        new_line = translate_line(line)
        # concatenate the new string and add an end-line break
        replaced_content = replaced_content + new_line + "\n"

    # close the file
    file.close()
    # Open file in write mode
    write_file = open(path, "w")
    # overwriting the old file contents with the new/replaced content
    write_file.write(replaced_content)
    # close the file
    write_file.close()


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-o", "--obj_dir", required=True, type=str)
    args = vars(ap.parse_args())

    labels = []
    for r, d, f in os.walk(args["obj_dir"]):
        for file in f:
            if '.txt' in file:
                labels.append(os.path.join(r, file))

    print("INFO: Found a total of", len(labels), "files to convert.")

    for label_file in labels:
        process_file(label_file)

    print("INFO: Process completed.")
