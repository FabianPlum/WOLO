import cv2
import numpy as np
import os


def advanced_patch_export(tracks, clip_path, frame_start, frame_end, output_folder,
                          header=None, export_every_nth_frame=0, input_res=128, output_res=128):
    print("\nINFO: RUNNING ADVANCED SAMPLE EXTRACTION\n")

    # now we can load the captured video file and display it
    cap = cv2.VideoCapture(clip_path)
    num_tracks = int((tracks.shape[1] - 1) / 2)

    print("Extracting samples from", clip_path, "...")

    # export all tracked frames from all tracks
    for frame_id in range(frame_start, frame_end):

        cap.set(1, frame_id)
        ret, frame_temp = cap.read()

        for track_id in range(num_tracks):

            if tracks[frame_id, track_id * 2 + 1] != 0:
                """
                # skip all but nth frames
                if frame_id % export_every_nth_frame != 0:
                    continue
                """

                marker_x = tracks[frame_id, (track_id + 1) * 2 - 1]
                marker_y = tracks[frame_id, (track_id + 1) * 2]

                print("Frame:", frame_id, " : ",
                      "X", marker_x, ",",
                      "Y", marker_y)


                if ret:
                    # first, create an empty image object to be filled with the ROI
                    # this is important in case the detection lies close to the edge
                    # where the ROI would go outside the image
                    track_input_img = np.zeros([input_res,
                                                input_res, 3], dtype=np.uint8)

                    true_min_x = marker_x - int(input_res / 2)
                    true_max_x = marker_x + int(input_res / 2)
                    true_min_y = frame_temp.shape[0] - marker_y - int(input_res / 2)
                    true_max_y = frame_temp.shape[0] - marker_y + int(input_res / 2)

                    min_x = max([0, true_min_x])
                    max_x = min([frame_temp.shape[1], true_max_x])
                    min_y = max([0, true_min_y])
                    max_y = min([frame_temp.shape[0], true_max_y])
                    # crop frame to detection and rescale
                    frame_cropped = frame_temp[min_y:max_y, min_x:max_x]

                    # place the cropped frame in the previously created empty image
                    x_min_offset = max([0, - true_min_x])
                    x_max_offset = min([input_res,
                                        input_res - (true_max_x - frame_temp.shape[1])])
                    y_min_offset = max([0, - true_min_y])
                    y_max_offset = min([input_res,
                                        input_res - (true_max_y - frame_temp.shape[0])])

                    print("Cropped image ROI:", x_min_offset, x_max_offset, y_min_offset, y_max_offset)
                    track_input_img[y_min_offset:y_max_offset, x_min_offset:x_max_offset] = frame_cropped

                    track_input_img = cv2.resize(track_input_img,
                                                 (output_res,
                                                  output_res))

                    # save out image, using the following convention:
                    # clip-name_frame_track.format
                    if header is not None:
                        track_name = header.split(",")[track_id * 2 + 1].split("_")[0]
                    else:
                        track_name = track_id
                    out_path = str(os.path.join(os.path.abspath(output_folder),
                                                os.path.basename(
                                                    clip_path[:-4]))) + "_" + str(
                        frame_id) + "_" + track_name + ".jpg"
                    print(out_path)

                    # now, write out the final patch to the desired location
                    cv2.imwrite(out_path, track_input_img, )

        print("\n")

    cv2.destroyAllWindows()

    # always reset frame from capture at the end to avoid incorrect skips during access
    cap.set(1, frame_start - 1)
    cap.release()
    print("Read all frames")

    return
