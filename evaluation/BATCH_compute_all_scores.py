import os
import subprocess

results_path = "I:/WOLO/BENCHMARK/RESULTS_TEST/CORVIN9000"
model_list = os.listdir(os.path.join(results_path, "predictions"))

for model in model_list:
    subprocess.call(["python",
                     "compute_all_scores.py",
                     "-i", str(os.path.join(os.path.join(results_path, "predictions"), model)),
                     "-o", results_path])
