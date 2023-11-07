import os
import subprocess

results_path = "I:/WOLO/BENCHMARK/RESULTS_VAL_DETECT_TEST"
model_path = "D:/WOLO/HPC_trained_models/WOLO_DETECT/OUTPUT"
model_list = os.listdir(model_path)

for model in model_list:
    subprocess.call(["python",
                     "compute_all_scores.py",
                     "-i", str(os.path.join(model_path, model)),
                     "-o", results_path])#,
                     #"-gt", "True"])  # required for CORVIN9000 data (DETECT only)
