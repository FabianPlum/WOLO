import os
from concurrent.futures import ProcessPoolExecutor as Executor
import compute_all_scores
import pandas as pd


def worker(task):
    # task = [model, requires_gt_extraction, detection]
    try:
        if not task[2]:
            return compute_all_scores.compute_scores(input_file=task[0],
                                                     output=task[3],
                                                     dataset_name=task[4])

        else:
            if task[1]:
                return compute_all_scores.compute_scores(input_file=task[0],
                                                         output=task[3],
                                                         detection_format=True,
                                                         gt_from_name=True,
                                                         dataset_name=task[4])

            else:
                return compute_all_scores.compute_scores(input_file=task[0],
                                                         output=task[3],
                                                         detection_format=True,
                                                         dataset_name=task[4])
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error)
        print("\n\n---------------\nERROR: CANNOT PROCESS", task[0], "\n\n---------------\n")


if __name__ == '__main__':
    results_path = "D:/WOLO/BENCHMARK/EVALUATION"

    # [path, special-weight-formatting, detection]

    all_file_paths = [["D:/WOLO/HPC_trained_models/WOLO/TRAINED_MODELS", False, False,
                       "VAL"],
                      ["D:/WOLO/BENCHMARK/RESULTS_TEST/CORVIN9000/predictions", False, False,
                       "CORVIN9000"],
                      ["D:/WOLO/BENCHMARK/RESULTS_TEST/DSLR_C920/predictions", False, False,
                       "DSLR-C920"],
                      ["D:/WOLO/HPC_trained_models/WOLO_DETECT/RESULTS", False, True,
                       "VAL"],
                      ["D:/WOLO/HPC_trained_models/WOLO_DETECT/RESULTS_TEST_128x128_DSLR_C920", False, True,
                       "DSLR-C920"],
                      ["D:/WOLO/HPC_trained_models/WOLO_DETECT/RESULTS_TEST_128x128_CORVIN9000", True, True,
                       "CORVIN9000"]]
    """

    all_file_paths = [["D:/WOLO/HPC_trained_models/WOLO_DETECT/RESULTS", False, True,
                       "VAL"]]
    """

    parallel = True

    # define pool_size and create worker pool
    all_results = {}

    if parallel:
        with Executor() as executor:

            for e, eval_type in enumerate(all_file_paths):
                print("SET", e, "of", eval_type)

                model_path = eval_type[0]

                model_list = os.listdir(model_path)

                for model in model_list:
                    all_results[model + "_data_" + eval_type[3]] = executor.submit(worker,
                                                                                   [str(os.path.join(model_path,
                                                                                                     model)),
                                                                                    eval_type[1],
                                                                                    eval_type[2],
                                                                                    results_path,
                                                                                    eval_type[3]])

        out_results = all_results.copy()
        for i, future in all_results.items():
            if future.result() is not None:
                print(i, "  :  ", future.result())
            else:
                del out_results[i]

        print("Out results:", len(out_results))


    else:
        for e, eval_type in enumerate(all_file_paths):
            print("SET", e, "of", eval_type)
            model_path = eval_type[0]

            model_list = os.listdir(model_path)

            for model in model_list:
                all_results[model + "_data_" + eval_type[3]] = worker([str(os.path.join(model_path, model)),
                                                                       eval_type[1],
                                                                       eval_type[2],
                                                                       results_path,
                                                                       eval_type[3]])

        print("\n\n================================================\n\n")

        out_results = all_results.copy()
        for i, result in all_results.items():
            if result is not None:
                print(i, "  :  ", result)
            else:
                del out_results[i]

        print("Out results:", len(out_results))

    print("\n\n================================================\n\n FINISHED")
