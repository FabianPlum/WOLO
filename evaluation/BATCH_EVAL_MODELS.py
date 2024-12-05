import os
import subprocess
import sys

# Define paths using os.path.join for better cross-platform compatibility
model_path = os.path.join("..", "TRAINED_NETWORKS")
datasets = [
    os.path.join("..", "DATASETS", "CORVIN9000"),
    os.path.join("..", "DATASETS", "MultiCamAnts", "test"),
    os.path.join("..", "DATASETS", "DSLR_C920_CLASS")
]
output_dir = os.path.join("..", "RESULTS-TEST")

# Check if model directory exists
if not os.path.exists(model_path):
    print(f"Error: Model directory not found: {model_path}")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of models
try:
    model_list = os.listdir(model_path)
except Exception as e:
    print(f"Error accessing model directory: {e}")
    sys.exit(1)

# Process each dataset and model
for dataset in datasets:
    if not os.path.exists(dataset):
        print(f"Warning: Dataset not found, skipping: {dataset}")
        continue
        
    for model in model_list:
        model_full_path = os.path.join(model_path, model)
        try:
            subprocess.run([
                "python",
                "EVALUATE_MODEL_ON_TEST_DATA.py",
                "-i", model_full_path,
                "-d", dataset,
                "-o", output_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating model {model} on dataset {dataset}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
