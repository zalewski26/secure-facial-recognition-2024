import argparse
import json
import time

import numpy as np

from utils.experiment_parameters import ExperimentParameters
from utils.similarity import evaluate_accuracy
from utils.utils import Model, Modification, get_embedding_path

RESULTS_PATH = "results.json"


def _save_results(new_results):
    try:
        with open(RESULTS_PATH, "r") as infile:
            results = json.load(infile)
    except FileNotFoundError:
        results = {}
    results.update(new_results)

    with open(RESULTS_PATH, "w") as outfile:
        json.dump(results, outfile, indent=4)


def evaluate(
    num_classes,
    db_images,
    test_images,
    model_name,
    db_modification,
    test_modification,
    db_mod_param,
    test_mod_param,
):
    evaluation_results = {}
    db_path = get_embedding_path(num_classes, db_images, test_images, model_name, db_modification, db_mod_param)
    test_path = get_embedding_path(num_classes, db_images, test_images, model_name, test_modification, test_mod_param)

    if db_modification is None:
        config_key = f"{num_classes}_{db_images}_{test_images}_{model_name}_clean"
    else:
        config_key = f"{num_classes}_{db_images}_{test_images}_{model_name}_{db_modification}_{db_mod_param}"
    if test_modification is None:
        config_key += "_clean"
    else:
        config_key += f"_{test_modification}_{test_mod_param}"

    thresholds = np.linspace(0, 1, 201)
    eer, acc_at_eer, eer_thresh, accuracies, fars, frrs, misses = evaluate_accuracy(db_path, test_path, thresholds)

    evaluation_results[config_key] = {}
    evaluation_results[config_key]["EER"] = round(eer, 3)
    evaluation_results[config_key]["Accuracy at EER"] = round(acc_at_eer, 3)
    evaluation_results[config_key]["EER Threshold"] = round(eer_thresh, 3)
    evaluation_results[config_key]["Thresholds"] = [round(t, 3) for t in thresholds]
    evaluation_results[config_key]["Accuracies"] = [round(acc, 3) for acc in accuracies]
    evaluation_results[config_key]["FARs"] = [round(far, 3) for far in fars]
    evaluation_results[config_key]["FRRs"] = [round(frr, 3) for frr in frrs]
    evaluation_results[config_key]["Misclassifications"] = [round(miss, 3) for miss in misses]

    _save_results(evaluation_results)


def evaluate_modifications(num_classes, db_images, test_images, model_name, db_mod, test_mod, db_mod_param):
    start = time.time()
    for test_modification in test_mod:
        test_mod_params = ExperimentParameters.MOD_PARAMS.get(test_modification, [None])
        for test_mod_param in test_mod_params:
            evaluate(
                num_classes,
                db_images,
                test_images,
                model_name,
                db_mod,
                test_modification,
                db_mod_param,
                test_mod_param,
            )
    end = time.time()
    print(
        f"Completed: {num_classes}_{db_images}_{test_images}_{model_name} ({db_mod or 'clean'}) ({(end - start):.3f}s)"
    )


def evaluate_all():
    for num_classes in ExperimentParameters.CLASSES_NUMBER:
        for db_images in ExperimentParameters.DB_IMAGES:
            for test_images in ExperimentParameters.TEST_IMAGES:
                for model in ExperimentParameters.MODELS:
                    model_name = str(model)

                    evaluate_modifications(
                        num_classes,
                        db_images,
                        test_images,
                        model_name,
                        None,
                        ExperimentParameters.MODIFICATIONS,
                        None,
                    )

                    for db_mod in ExperimentParameters.MODIFICATIONS:
                        if db_mod is None:
                            continue
                        db_mod_params = ExperimentParameters.MOD_PARAMS[db_mod]
                        for db_mod_param in db_mod_params:
                            evaluate_modifications(
                                num_classes,
                                db_images,
                                test_images,
                                model_name,
                                db_mod,
                                [None, db_mod],
                                db_mod_param,
                            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate embeddings and save results in accuracy.json.")
    parser.add_argument(
        "--mode", type=str, choices=["all", "one"], default="one", help="Run mode: all configurations or one."
    )
    parser.add_argument("--num_classes", type=int, help="Number of classes.")
    parser.add_argument("--database_images", type=int, help="Number of images in database per class.")
    parser.add_argument("--test_images", type=int, help="Number of test images per class.")
    parser.add_argument("--model", type=str, default="FACENET", choices=[m.name for m in Model], help="Model name.")
    parser.add_argument(
        "--modification",
        type=str,
        default=None,
        choices=[None] + [m.name for m in Modification],
        help="Type of modification applied to images.",
    )
    parser.add_argument("--mod_param", type=str, help="Parameter for modification e.g. mode for Fawkes.")

    args = parser.parse_args()

    if args.mode == "all":
        evaluate_all()
    else:
        model_name = str(Model[args.model])
        modification = Modification[args.modification] if args.modification else None
        evaluate(args.num_classes, args.database_images, args.test_images, model_name, modification, args.mod_param)


if __name__ == "__main__":
    main()
