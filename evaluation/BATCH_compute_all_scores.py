import os
import csv
from concurrent.futures import ProcessPoolExecutor as Executor
import compute_all_scores


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

    parallel = True

    # define pool_size and create worker pool
    all_results = []

    if parallel:
        with Executor() as executor:

            for e, eval_type in enumerate(all_file_paths):
                print("SET", e, "of", eval_type)

                model_path = eval_type[0]

                model_list = os.listdir(model_path)

                for model in model_list:
                    all_results.append(executor.submit(worker,
                                                       [str(os.path.join(model_path,
                                                                         model)),
                                                        eval_type[1],
                                                        eval_type[2],
                                                        results_path,
                                                        eval_type[3]]))

        out_results = []
        for i, future in enumerate(all_results):
            result = future.result()
            if result is not None:
                print(i, "  :  ", result)
                out_results.append(result)

        print("Out results:", len(out_results))


    else:
        for e, eval_type in enumerate(all_file_paths):
            print("SET", e, "of", eval_type)
            model_path = eval_type[0]

            model_list = os.listdir(model_path)

            for model in model_list:
                all_results.append(worker([str(os.path.join(model_path, model)),
                                           eval_type[1],
                                           eval_type[2],
                                           results_path,
                                           eval_type[3]]))

        print("\n\n================================================\n\n")

        out_results = []
        for i, result in enumerate(all_results):
            if result is not None:
                print(i, "  :  ", result)
                out_results.append(result)

        print("Out results:", len(out_results))

    """
    Now, sort the retrieved results into dicts to dump into a csv file:
                                                ________________________(test) dataset_______________________________
                attributes                      VAL                     DSLR-C920               CORVIN9000
    model       real_data   synt_data   ...     R^2     MAPE    COV     R^2     MAPE    COV     R^2     MAPE    COV
    model_XYZ   ####        ####        ...     ####    ####    ####    ####    ####    ####    ####    ####    #### 
    
    all attributes
    "model"
    "dataset"           -> VAL / DSLR-C920 / CORVIN9000
    "inference_type"    -> REG / CLASS / DETECT
    "real data"         -> MultiCamAnts
    "synth data"        -> simple / standard / all
    "sigma"             -> one-hote / 0.5 / 1.0 / 2.0 / 4.0
    "R^2" [log]  
    "slope"
    "y intercept"
    "accuracy"
    "MAPE_true"
    "MAPE_ideal"
    "MAPE_class"
    "COV"
    """

    attributes = ["model", "dataset", "inference_type", "real data", "synth data", "sigma"]

    out_clean = {}
    for result in out_results:
        temp_entry = {}
        # if the model is already part of the dictionary, only add results
        if result["model"] in out_clean:
            for key, value in result.items():
                if key not in attributes:
                    out_clean[result["model"]][result["dataset"] + "_" + key] = value

        # if the model appears for the first time, add a new entry
        else:
            # first, get all static attributes
            for key in attributes:
                # don't write out (test) dataset
                if key != "dataset":
                    temp_entry[key] = result[key]
            # then, get all test-data specific results
            for key, value in result.items():
                if key not in attributes:
                    temp_entry[result["dataset"] + "_" + key] = value

            out_clean[result["model"]] = temp_entry

    print(len(out_clean))

    with open(os.path.join(results_path, 'all_results.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, out_clean[list(out_clean.keys())[0]].keys())
        w.writeheader()
        w.writerows(list(out_clean.values()))

    print("\n\n================================================\n\n FINISHED")
