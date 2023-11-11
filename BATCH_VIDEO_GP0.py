import os
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
              "2019-08-06/2019-08-06_bramble_left.avi"]

video_absolute_path = "I:/EAEAAO/FOOTAGE"
tracks_absolute_path = "J:/OUTPUT_TRACKS"

model_path = "D:/WOLO/HPC_trained_models/WOLO/TRAINED_MODELS/CLASS_MultiCamAnts-and-synth-standard_20_one-hot_cross-entropy"

for video in video_list:
    video_path = os.path.join(video_absolute_path, video)
    tracks_path = os.path.join(tracks_absolute_path, video.split("/")[-1][:-4])

    if os.path.exists(video_path):
        print("INFO: ", video_path, "checks out!")
    else:
        print("INFO: ", video_path, "DOES NOT CHECK OUT!!!")

    if os.path.exists(tracks_path):
        print("INFO: ", tracks_path, "checks out!")
    else:
        print("INFO: ", tracks_path, "DOES NOT CHECK OUT!!!")

    print("INFO: Processing", video)

    subprocess.call(["python",
                     "extract_size_frequency_from_video.py",
                     "-v", str(video_path),
                     "-t", str(tracks_path),
                     "-m", model_path,
                     "--GPU", "0"])
                     #"-d", "False"])
