import json

import numpy as np

from utils.experiment_parameters import ExperimentParameters
from utils.similarity import evaluate_sensitivity
from utils.utils import get_embedding_path

RESULTS_PATH = "sensitivity.json"


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
    modification,
    mod_param,
):
    evaluation_results = {}
    db_path = get_embedding_path(num_classes, db_images, test_images, model_name, modification, mod_param)
    test_path = get_embedding_path(num_classes, db_images, test_images, model_name, modification, mod_param)

    if modification is None:
        config_key = f"{num_classes}_{db_images}_{test_images}_{model_name}_clean"
    else:
        config_key = f"{num_classes}_{db_images}_{test_images}_{model_name}_{modification}_{mod_param}"

    thresholds = np.linspace(0, 1, 201)
    base_acc, distant_acc, sensitivity = evaluate_sensitivity(db_path, test_path, thresholds)

    evaluation_results[config_key] = {}
    evaluation_results[config_key]["Base Accuracy"] = round(base_acc, 4)
    evaluation_results[config_key]["Distant Accuracy"] = round(distant_acc, 4)
    evaluation_results[config_key]["Sensitivity"] = round(sensitivity, 4)

    _save_results(evaluation_results)


def evaluate_all():
    for num_classes in ExperimentParameters.CLASSES_NUMBER:
        for db_images in ExperimentParameters.DB_IMAGES:
            for test_images in ExperimentParameters.TEST_IMAGES:
                for model in ExperimentParameters.MODELS:
                    model_name = str(model)
                    for mod in ExperimentParameters.MODIFICATIONS:
                        mod_params = ExperimentParameters.MOD_PARAMS.get(mod, [None])
                        for mod_params in mod_params:
                            evaluate(num_classes, db_images, test_images, model_name, mod, mod_params)


def main():
    evaluate_all()


if __name__ == "__main__":
    main()
