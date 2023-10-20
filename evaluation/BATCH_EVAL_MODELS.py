import os
import subprocess

model_path = "Z:/home/WOLO/TRAINED-NETWORKS"
datasets = ["I:/WOLO/BENCHMARK/DSLR_C920_CLASS",
            "I:/WOLO/BENCHMARK/CORVIN9000/CORVIN9000"]
output_dir = "I:/WOLO/BENCHMARK/RESULTS_TEST"

model_list = os.listdir(model_path)

for dataset in datasets:
    for model in model_list:
        subprocess.call(["python",
                         "EVALUATE_MODEL_ON_TEST_DATA.py",
                         "-i", str(os.path.join(model_path, model)),
                         "-d", dataset,
                         "-o", output_dir])
