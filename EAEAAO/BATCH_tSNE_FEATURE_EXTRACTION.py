import os
from multiprocessing.pool import ThreadPool
import subprocess

video_list = ["2019-07-22/2019-07-22_bramble_left.mp4",
              "2019-07-23/2019-07-23_bramble_right2.avi",
              "2019-07-23/2019-07-23_bramble_right.avi",
              "2019-07-23/2019-07-23_rose_left_2.avi",
              "2019-07-23/2019-07-23_rose_left.avi",
              "2019-07-24/2019-07-24_bramble_left.avi",
              "2019-07-24/2019-07-24_bramble_right.avi",
              "2019-07-25/2019-07-25_rose_left.avi",
              "2019-07-25/2019-07-25_rose_right.avi",
              "2019-07-30/2019-07-30_bramble_left.avi",
              "2019-07-30/2019-07-30_rose_right.avi",
              "2019-07-31/2019-07-31_bramble_left.avi",
              "2019-07-31/2019-07-31_bramble_right.avi",
              "2019-08-01/2019-08-01_bramble_left.avi",
              "2019-08-01/2019-08-01_rose_right.avi",
              "2019-08-03/2019-08-03_bramble-left.avi",
              "2019-08-03/2019-08-03_bramble-right.avi",
              "2019-08-05/2019-08-05_bramble_left.avi",
              "2019-08-05/2019-08-05_rose_right.avi",
              "2019-08-06/2019-08-06_bramble_left.avi",
              "2019-08-06/2019-08-06_rose_right.avi",
              "2019-08-07/2019-08-07_bramble_left.avi",
              "2019-08-07/2019-08-07_bramble_right.avi",
              "2019-08-08/2019-08-08_rose_left.avi",
              "2019-08-08/2019-08-08_rose_right.avi",
              "2019-08-09/2019-08-09_bramble_left.avi",
              "2019-08-09/2019-08-09_rose_right.avi",
              "2019-08-12/2019-08-12_rose_left.avi",
              "2019-08-12/2019-08-12_rose_right.avi",
              "2019-08-13/2019-08-13_bramble_right.avi",
              "2019-08-13/2019-08-13_rose_left.avi",
              "2019-08-15/2019-08-15_bramble_left.avi",
              "2019-08-15/2019-08-15_bramble_right.avi",
              "2019-08-16/2019-08-16_bramble_right.avi",
              "2019-08-16/2019-08-16_rose_left.avi",
              "2019-08-20/2019-08-20_rose_left.avi",
              "2019-08-20/2019-08-20_rose_right.avi",
              "2019-08-21/2019-08-21_rose_left.avi",
              "2019-08-21/2019-08-21_rose_right.avi",
              "2019-08-22/2019-08-22_bramble_right.avi",
              "2019-08-22/2019-08-22_rose_left.avi"]

order_list = ["I:/EAEAAO/jobscripts/EAEAAO_batch_pose.pbs.o8454338." + str(i + 1) for i in range(len(video_list))]

VIDEO_FOLDER = "I:/EAEAAO/FOOTAGE/"
TRACK_FOLDER = "I:/EAEAAO/Tracking/OUTPUT_TRACKS/"
POSE_FOLDER = "I:/EAEAAO/POSES/"


def work(task):
    input_video = VIDEO_FOLDER + task[0]
    input_tracks = TRACK_FOLDER + task[0].split("/")[-1][:-4]
    input_poses = POSE_FOLDER + task[0].split("/")[-1][:-4]
    order = task[1]
    feature_extract_subprocess = subprocess.call(["python",
                                                  "tSNE_feature_extractor.py",
                                                  "-t", input_tracks,
                                                  "-v", input_video,
                                                  "-p", input_poses,
                                                  "-o", order])

    # "-test", str(True)])

    line = True
    while line:
        parsedline = feature_extract_subprocess.stdout.readline()
        print(parsedline)


num = 8  # set to the number of workers you want (it defaults to the cpu count of your machine)
tp = ThreadPool(num)
for video, order in zip(video_list, order_list):
    tp.apply_async(work, ([video, order],))

tp.close()
tp.join()
