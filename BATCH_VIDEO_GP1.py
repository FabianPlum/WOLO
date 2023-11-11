import os
import subprocess

video_list = ["2019-08-06/2019-08-06_rose_right.avi",
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

video_absolute_path = "I:/EAEAAO/FOOTAGE"
tracks_absolute_path = "J:/OUTPUT_TRACKS"

# CLASS_MultiCamAnts-and-synth-simple_5_sigma-2_cross-entropy
# CLASS_MultiCamAnts-and-synth-standard_20_one-hot_cross-entropy
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
                     "--GPU", "1"])
                     #"-d", "True"])
