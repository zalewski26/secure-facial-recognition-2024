import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from utils.experiment_parameters import ExperimentParameters
from utils.similarity import load_and_prepare_embeddings
from utils.utils import Model, Modification, get_embedding_path

RESULTS_PATH = "differential_privacy.json"


def _save_results(new_results):
    try:
        with open(RESULTS_PATH, "r") as infile:
            results = json.load(infile)
    except FileNotFoundError:
        results = {}
    results.update(new_results)

    with open(RESULTS_PATH, "w") as outfile:
        json.dump(results, outfile, indent=4)


def calc_probabilities(db_embeddings, test_embeddings, threshold):
    sim_matrix = np.dot(test_embeddings, db_embeddings.transpose((0, 2, 1)))
    max_emb_value = np.max(sim_matrix, axis=-1)
    max_class_value = np.max(max_emb_value, axis=-1)
    accepted = np.sum(max_class_value > threshold, axis=-1)
    result = [accepted[j] / test_embeddings.shape[1] for j in range(test_embeddings.shape[0])]
    return result


def calculate_epsilon(original_probs, mod_probs):
    epsilons = []
    for p_d, p_d_prime in zip(original_probs, mod_probs):
        if p_d_prime > 0:
            epsilon = abs(np.log(p_d / p_d_prime))
            epsilons.append(epsilon)
    
    return max(epsilons) if epsilons else float('inf')


def evaluate(num_classes, db_images, test_images, model_name, modification, mod_param, threshold):
    db_path = get_embedding_path(num_classes, db_images, test_images, model_name, modification, mod_param)
    test_path = get_embedding_path(num_classes, db_images, test_images, model_name, modification, mod_param)
    if modification is None:
        config_key = f"{num_classes}_{db_images}_{test_images}_{model_name}_clean"
    else:
        config_key = f"{num_classes}_{db_images}_{test_images}_{model_name}_{modification}_{mod_param}"

    db_embeddings = load_and_prepare_embeddings(os.path.join(db_path, "database.pkl"))
    test_embeddings = load_and_prepare_embeddings(os.path.join(test_path, "test_known.pkl"))
    
    all_original_probs = []
    all_mod_probs = []
    all_epsilons = []

    for i in tqdm(range(db_embeddings.shape[0])):
        mod_db_embeddings = np.delete(db_embeddings, i, axis=0)
        
        original_probs = calc_probabilities(db_embeddings, test_embeddings, threshold)
        mod_probs = calc_probabilities(mod_db_embeddings, test_embeddings, threshold)
        epsilon = calculate_epsilon(original_probs, mod_probs)
        
        all_original_probs.append(original_probs)
        all_mod_probs.append(mod_probs)
        all_epsilons.append(epsilon)
    
    max_epsilon_index = np.argmax(all_epsilons)
    max_epsilon = all_epsilons[max_epsilon_index]
    eps_original_probs = all_original_probs[max_epsilon_index]
    eps_mod_probs = all_mod_probs[max_epsilon_index]
    
    evaluation_results = {
        config_key: {
            "Index": int(max_epsilon_index),
            "Epsilon": round(max_epsilon, 4),
            "x' original": eps_original_probs[max_epsilon_index],
            "x' mod": eps_mod_probs[max_epsilon_index],
            "Max originals": max(np.delete(eps_original_probs, max_epsilon_index)),
            "Min originals": min(np.delete(eps_original_probs, max_epsilon_index)),
            "Avg originals": np.mean(np.delete(eps_original_probs, max_epsilon_index)),
            "Max mod": max(np.delete(eps_mod_probs, max_epsilon_index)),
            "Min mod": min(np.delete(eps_mod_probs, max_epsilon_index)),
            "Avg mod": np.mean(np.delete(eps_mod_probs, max_epsilon_index)),
            
        }
    }
    _save_results(evaluation_results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate embeddings and save results in accuracy.json.")
    parser.add_argument(
        "--mode", type=str, choices=["all", "one"], default="one", help="Run mode: all configurations or one."
    )
    parser.add_argument("--num_classes", type=int, help="Number of classes.")
    parser.add_argument("--database_images", default=10, type=int, help="Number of images in database per class.")
    parser.add_argument("--test_images", default=30, type=int, help="Number of test images per class.")
    parser.add_argument("--model", type=str, choices=[m.name for m in Model], help="Model name.")
    parser.add_argument(
        "--modification",
        type=str,
        default=None,
        choices=[None] + [m.name for m in Modification],
        help="Type of modification applied to images.",
    )
    parser.add_argument("--mod_param", default=None, type=str, help="Parameter for modification e.g. mode for Fawkes.")
    parser.add_argument("--threshold", default=0.625, type=float)

    args = parser.parse_args()

    model_name = str(Model[args.model])
    modification = Modification[args.modification] if args.modification else None
    evaluate(
        args.num_classes,
        args.database_images,
        args.test_images,
        model_name,
        modification,
        args.mod_param,
        args.threshold,
    )


if __name__ == "__main__":
    main()
