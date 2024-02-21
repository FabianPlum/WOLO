import os
import concurrent
import compute_all_scores


def worker(task):
    # task = [model, requires_gt_extraction, detection]
    try:
        if not task[2]:
            compute_all_scores.compute_scores(input_file=task[0],
                                              output=task[3])

        else:
            if task[1]:
                compute_all_scores.compute_scores(input_file=task[0],
                                                  output=task[3],
                                                  detection_format=True,
                                                  gt_from_name=True)

            else:
                compute_all_scores.compute_scores(input_file=task[0],
                                                  output=task[3],
                                                  detection_format=True)
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error)
        print("\n\n---------------\nERROR: CANNOT PROCESS", task[0], "\n\n---------------\n")


if __name__ == '__main__':
    results_path = "D:/WOLO/BENCHMARK/EVALUATION"

    # [path, special-weight-formatting, detection]

    all_file_paths = [["D:/WOLO/HPC_trained_models/WOLO_DETECT/RESULTS", False, True],
                      ["D:/WOLO/HPC_trained_models/WOLO_DETECT/RESULTS_TEST_128x128_DSLR_C920", False, True],
                      ["D:/WOLO/HPC_trained_models/WOLO_DETECT/RESULTS_TEST_128x128_CORVIN9000", True, True],
                      ["D:/WOLO/HPC_trained_models/WOLO/TRAINED_MODELS", False, False],
                      ["D:/WOLO/BENCHMARK/RESULTS_TEST/CORVIN9000/predictions", False, False],
                      ["D:/WOLO/BENCHMARK/RESULTS_TEST/DSLR_C920/predictions", False, False]]

    # define pool_size and create worker pool
    executor = concurrent.futures.ProcessPoolExecutor(10)
    futures = []

    for eval_type in all_file_paths:
        model_path = eval_type[0]

        model_list = os.listdir(model_path)

        for model in model_list:
            print(model)
            futures.append(
                executor.submit(worker,
                                [str(os.path.join(model_path, model)), eval_type[1], eval_type[2], results_path]))

    concurrent.futures.wait(futures)

    print("\n\n================================================\n\n FINISHED")
