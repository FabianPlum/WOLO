import os
import subprocess

results_path = "I:/WOLO/BENCHMARK/RESULTS_TEST_DETECT/128x128/CORVIN9000"
model_path = "I:/WOLO/BENCHMARK/RESULTS_TEST_DETECT/128x128/CORVIN9000/predictions"
model_list = os.listdir(model_path)

for model in model_list:
    subprocess.call(["python",
                     "compute_all_scores.py",
                     "-i", str(os.path.join(model_path, model)),
                     "-o", results_path,"-gt", "True"])  # required for CORVIN9000 data (DETECT only)
