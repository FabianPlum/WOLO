import numpy as np
import cv2
import csv
import os
import math
import time
import align_tracking_views
import patch_export
import YOLO_export
from os import path


def import_tracks(path, numFrames, export=False, fill_track_gaps=True):
    """
    Import all tracked paths (using blender motionExport.py) from specified folder and join them to a single array.
    Optionally, allows for export of created array containing all tracks into single .csv file

    :param path: location of exported .csv tracks
    :param numFrames: number of total analysed frames
    :param export: boolean, writes .csv file of all combined tracks if True

    :return: array of all imported tracks, row: frames, columns X / Y coordinates of individual track.
             The first column consists of the frame numbers for easier readability if exported as a single file.
    """
    print("[INFO]      importing tracks from :", path, "\n")
    files = []
    header = "frame,"  # create string of track names to write out into columns
    tracks = np.empty([numFrames + 1, 1])  # create array for all tracks
    tracks[:, 0] = np.arange(start=1, stop=numFrames + 2, step=1, dtype=int)  # insert frame numbers

    imported = 0

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))

                # for each new track create two "zeros" columns
                # zeros are handled as nonexistent instances
                tracks = np.append(tracks, np.zeros([numFrames + 1, 2]), axis=1)
                track_name = file.split("_")[-1][:-4]
                header += track_name + "_x," + track_name + "_y,"

                with open(files[imported]) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=';')
                    line_count = 0

                    next(csv_reader, None)  # skip the headers

                    for row in csv_reader:
                        tracks[int(row[0]) - 1, imported * 2 + 1] = int(row[1])
                        tracks[int(row[0]) - 1, imported * 2 + 2] = int(row[2])

                        line_count += 1

                    # now clean up the track, by filling any gaps
                    if fill_track_gaps:
                        last_valid = [0, 0]
                        for coord in tracks:
                            if coord[imported * 2 + 1] == 0 or coord[imported * 2 + 2] == 0:
                                coord[imported * 2 + 1] = last_valid[0]
                                coord[imported * 2 + 2] = last_valid[1]
                            else:
                                last_valid[0] = coord[imported * 2 + 1]
                                last_valid[1] = coord[imported * 2 + 2]

                    print("[INFO]      imported", str(file), f' with {line_count} points.')

                imported += 1

    tracks = tracks.astype(int)
    if export:
        np.savetxt("all_tracks.csv", tracks, delimiter=",", fmt="%d", header=header)

    print("\n[INFO]      Successfully combined the tracks of", imported, "individuals for training and display!")
    return tracks, header


def display_video(cap, tracks, show=(0, math.inf), scale=1.0, video_output_name=None, header=None):
    """
    Function displays imported footage with tracking results as overlay

    :param cap: Imported video file
    :param tracks: all imported tracks as a single array, created with import_tracks
    :param show: tuple of desired displayed frames
    :param scale: single float to up- or downscale resolution of display
    """
    tracks = (scale * tracks).astype(int)  # rescale pixel values of tracks
    # frame counter
    frame_num = show[0]

    # define the size of each tracking rectangle
    target_size = 200 * scale

    # get frame rate of imported footage
    fps = cap.get(cv2.CAP_PROP_FPS)

    # fix the seed for the same set of randomly assigned colours for each track
    np.random.seed(seed=0)
    colours = np.random.randint(low=0, high=255, size=((math.floor(((tracks.shape[1]) - 1) / 2)), 3))

    print("\n[INFO]      Displaying tracked footage!\n[INFO]      press 'q' to end display")

    # skip to desired start frame
    # Property identifier of cv2.CV_CAP_PROP_POS_FRAMES is 1, thus the first entry is 1
    cap.set(1, show[0])

    # set font from info display on frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    # if video_output_name is not None:
    if os.path.exists(video_output_name):
        print("[WARNING]   the file", video_output_name, "already exists and will NOT be overwritten!")
        video_output_name = None
    else:
        out = cv2.VideoWriter(video_output_name, cv2.VideoWriter_fourcc(*'XVID'), 23.975, (1920, 1080))

    while True:  # run until no more frames are available
        time_prev = time.time()
        # return single frame (ret = boolean, frame = image)
        ret, frame = cap.read()
        if not ret:
            break

        # scale down the video
        new_height = int(np.shape(frame)[0] * scale)
        new_width = int(np.shape(frame)[1] * scale)
        frame = cv2.resize(frame, (new_width, new_height))

        # iterate through all columns and draw rectangles for all non 0 values
        for track in range(math.floor(((tracks.shape[1]) - 1) / 2)):
            if tracks[frame_num, track * 2 + 1] != 0:
                # the tracks are read as centres
                target_centre = np.asarray([tracks[frame_num, track * 2 + 1], tracks[frame_num, track * 2 + 2]])

                # invert y-axis, to fit openCV convention ( lower left -> (x=0,y=0) )
                target_centre[1] = new_height - target_centre[1]
                # define the starting and ending point of the bounding box rectangle, defined by "target_size"
                px_start = target_centre - np.asarray([math.floor(target_size / 2), math.floor(target_size / 2)])
                px_end = target_centre + np.asarray([math.floor(target_size / 2), math.floor(target_size / 2)])
                # draw the defined rectangle of the track on top of the frame
                cv2.rectangle(frame, (px_start[0], px_start[1]), (px_end[0], px_end[1]),
                              (int(colours[track, 0]), int(colours[track, 1]), int(colours[track, 2])), 2)
                # write out track number of each active track
                if header is not None:
                    track_name = header.split(",")[track * 2 + 1].split("_")[0]
                else:
                    track_name = "track: " + str(track)
                cv2.putText(frame, track_name,
                            (int(target_centre[0] - target_size / 2), int(target_centre[1] - target_size / 2 - 10)),
                            font, 0.3, (int(colours[track, 0]), int(colours[track, 1]), int(colours[track, 2])), 1,
                            cv2.LINE_AA)

        cv2.putText(frame, "frame: " + str(frame_num), (int(new_width / 2) - 100, 35),
                    font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('original frame', frame)

        if frame_num > show[1]:
            break

        # enforce constant frame rate during display
        time_to_process = (time.time() - time_prev)  # compute elapsed time to enforce constant frame rate (if possible)
        if time_to_process < 1 / fps:
            time.sleep((1 / fps) - time_to_process)

        # press q to quit, i.e. exit the display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

        if video_output_name is not None:
            out.write(frame)

    if video_output_name is not None:
        out.release()

    cv2.destroyAllWindows()

    # always reset frame from capture at the end to avoid incorrect skips during access
    cap.set(1, 0)

    print("\n[INFO]      Reached last frame of specified video or ended by user input.\n")


def translate_tracks(refCap, targetCap, tracks, useFrame=100, header=None):
    refCap.set(1, useFrame)
    ret, refFrame = refCap.read()

    targetCap.set(1, useFrame)
    ret, targetFrame = targetCap.read()

    print("[INFO]      Aligning frames...")
    # Registered image will be restored in imReg.
    # The estimated homography will be stored in h
    imReg, h = align_tracking_views.alignImages(refFrame, targetFrame)
    print("[INFO]      Frames aligned!\n")

    num_tracks = int((tracks.shape[1] - 1) / 2)

    target_tracks = tracks.copy()

    height = refFrame.shape[0]
    width = refFrame.shape[1]

    target_height = targetFrame.shape[0]
    target_width = targetFrame.shape[1]

    print("Height:", height)
    print("Width:", width)

    print("\nTarget Height:", target_height)
    print("Target Width:", target_width)

    print("\nTransformation matrix from homography:\n", h, "\n")

    last_valid = np.zeros((num_tracks, 2))

    for track in range(num_tracks):
        for frame in range(tracks.shape[0]):
            if tracks[frame, (track + 1) * 2 - 1] == 0 or tracks[frame, (track + 1) * 2] == 0:
                target_tracks[frame, (track + 1) * 2 - 1] = last_valid[track][0]
                target_tracks[frame, (track + 1) * 2] = last_valid[track][1]
            else:
                x = float(tracks[frame, (track + 1) * 2 - 1])
                # cause blender uses an inverted coordinate system... took me 24 hours and Lea's intuition to realise
                y = height - float(tracks[frame, (track + 1) * 2])

                dst_x = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / (h[2, 0] * x + h[2, 1] * y + h[2, 2])
                dst_y = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / (h[2, 0] * x + h[2, 1] * y + h[2, 2])

                last_valid[track] = [dst_x, abs(dst_y - target_height)]

                target_tracks[frame, (track + 1) * 2 - 1] = dst_x
                target_tracks[frame, (track + 1) * 2] = abs(dst_y - target_height)

    target_tracks = target_tracks.astype(int)
    np.savetxt("target_tracks.csv", target_tracks, delimiter=",", fmt="%d", header=header)

    return target_tracks, h


def exportWarpedVideo(cap, h, output_name='warped_video.avi'):
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'XVID'), 23.975, (1920, 1080))
    cap.set(1, 0)
    while True:  # run until no more frames are available
        # return single frame (ret = boolean, frame = image)
        ret, frame = cap.read()
        if not ret:
            break

        warpedFrame = cv2.warpPerspective(frame, h, (1920, 1080))

        out.write(warpedFrame)

    out.release()


if __name__ == "__main__":
    np.random.seed(seed=0)

    path = 'RAW\\OAK-D\\BROWN_FOREST_FLOOR_2023-04-10_16-05-32-05S'
    # load reference (tracked) video
    refFile = "RAW\\OAK-D\\BROWN_FOREST_FLOOR_2023-04-10_16-05-32-05S.mp4"
    print("[INFO]      loading reference video :", refFile.split("\\")[-1])
    refCap = cv2.VideoCapture(refFile)

    # load input video (transfer tracks onto this clip)
    targetFile = "processed\\BROWN_FOREST_C920_SYNCHRONISED.mp4"
    print("[INFO]      loading target video :", targetFile.split("\\")[-1])
    targetCap = cv2.VideoCapture(targetFile)

    numFramesMax = int(refCap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks, header = import_tracks(path, numFramesMax, export=True)

    frame_start = 4111

    """
    patch_export.advanced_patch_export(tracks=tracks, clip_path=refFile,
                                       frame_start=frame_start,
                                       frame_end=frame_start+10000,
                                       output_folder="MultiCamAnts_sub-datasets/DRY-LEAVES_OAK-D_patches_new", header=header,
                                       export_every_nth_frame=0, input_res=128, output_res=128)
    """

    # align two frames from the synchronised videos using homography to obtain the required rotation matrix
    translated_tracks, h = translate_tracks(refCap, targetCap, tracks, useFrame=6000, header=header)


    """
    patch_export.advanced_patch_export(tracks=translated_tracks, clip_path=targetFile,
                                       frame_start=frame_start,
                                       frame_end=frame_start + 10000,
                                       output_folder="MultiCamAnts_sub-datasets/BROWN-FOREST_C920_patches", header=header,
                                       export_every_nth_frame=0, input_res=100, output_res=128)
    """

    """
    # export transformed video:
    exportWarpedVideo(refCap, h, output_name='warped_' + refFile.split('.')[0] + '.avi')
    """

    # display video with track overlay
    display_video(refCap, tracks, show=(frame_start, frame_start + 500), scale=1,
                  video_output_name="orig_tracked.avi",
                  header=header)

    display_video(targetCap, translated_tracks, show=(frame_start, frame_start + 500), scale=1,
                  video_output_name="transfer_tracked.avi",
                  header=header)

    # release video cap at the end of the script and close all windows
    refCap.release()
    targetCap.release()

    cv2.destroyAllWindows()
