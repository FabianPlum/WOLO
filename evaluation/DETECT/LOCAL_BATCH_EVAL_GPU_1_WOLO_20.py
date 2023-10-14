import subprocess

# list all models that are to be evaluated by this thread

dataset = "I:\\WOLO\\BENCHMARK\\MultiCamAnts_YOLO\\data\\obj_test"


model_folder = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\TRAIN\\20_CLASS\\"

models = ["DETECT_MultiCamAnts_20",
          "DETECT_MultiCamAnts-and-synth-all_20",
          "DETECT_MultiCamAnts-and-synth-simple_20",
          "DETECT_MultiCamAnts-and-synth-standard_20",
          "DETECT_synth-all_20",
          "DETECT_synth-simple_20",
          "DETECT_synth-standard_20"]

cfg = "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\yolov4_20_class_WOLO_TEST.cfg"
meta = "I:\\WOLO\\BENCHMARK\\MultiCamAnts_YOLO\\data\\obj20.data"

for model in models:
    subprocess.call(["python", "darknet_evaluation_main.py",
                     "--modelFolder", model_folder + model,
                     "--dataFolder", dataset,
                     "--darknetFolder", "I:\\BENCHMARK\\DARKNET_TRAIN\\darknet\\x64",
                     "--configPath", cfg,
                     "--metaPath", meta,
                     "--outputFolder", "D:\\WOLO\\HPC_trained_models\\WOLO_DETECT\\OUTPUT\\" + model,
                     "--GPU", "1",
                     "--lastOnly", "True"])
