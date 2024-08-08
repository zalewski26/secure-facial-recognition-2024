import os
import pickle

import numpy as np


def load_and_prepare_embeddings(embedding_path):
    with open(embedding_path, "rb") as f:
        dict_embeddings = pickle.load(f)
    embeddings = np.stack([np.stack(embeds) for embeds in dict_embeddings.values()])
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def evaluate_accuracy(db_emb_path, test_emb_path, thresholds):
    db_embeddings = load_and_prepare_embeddings(os.path.join(db_emb_path, "database.pkl"))
    test_known_embeddings = load_and_prepare_embeddings(os.path.join(test_emb_path, "test_known.pkl"))
    test_unknown_embeddings = load_and_prepare_embeddings(os.path.join(test_emb_path, "test_unknown.pkl"))

    # shape (10, 30, 10, 10)
    known_sim_matrix = np.dot(test_known_embeddings, db_embeddings.transpose((0, 2, 1)))
    known_max_emb_value = np.max(known_sim_matrix, axis=-1)
    known_max_class_value = np.max(known_max_emb_value, axis=-1)
    known_max_class_index = np.argmax(known_max_emb_value, axis=-1)

    unknown_sim_matrix = np.dot(test_unknown_embeddings, db_embeddings.transpose((0, 2, 1)))
    unknown_max_emb_value = np.max(unknown_sim_matrix, axis=-1)
    unknown_max_class_value = np.max(unknown_max_emb_value, axis=-1)

    fars, frrs, accuracies, misses = [], [], [], []
    total_known = test_known_embeddings.shape[0] * test_known_embeddings.shape[1]
    total_unknown = test_unknown_embeddings.shape[0] * test_unknown_embeddings.shape[1]
    for threshold in thresholds:
        mask = known_max_class_value > threshold
        detected = np.sum(np.sum(mask))
        indices_above_thresh = np.copy(known_max_class_index)
        indices_above_thresh[~mask] = -1
        labels = np.arange(indices_above_thresh.shape[0])[:, np.newaxis]
        correct_predictions = np.sum(indices_above_thresh == labels)
        false_rejects = indices_above_thresh.size - detected
        wrong_user = detected - correct_predictions

        mask = unknown_max_class_value > threshold
        false_accepts = np.sum(np.sum(mask))
        correct_predictions += mask.size - np.sum(np.sum(mask))

        accuracy = correct_predictions / (total_known + total_unknown)
        far = false_accepts / total_unknown
        frr = false_rejects / total_known
        miss = wrong_user / total_known

        accuracies.append(accuracy)
        fars.append(far)
        frrs.append(frr)
        misses.append(miss)

    abs_differences = np.abs(np.array(fars) - np.array(frrs))
    min_index = np.argmin(abs_differences)
    eer = (fars[min_index] + frrs[min_index]) / 2
    acc_at_err = accuracies[min_index]
    eer_thresh = thresholds[min_index]
    return eer, acc_at_err, eer_thresh, accuracies, fars, frrs, misses


def evaluate_sensitivity(db_emb_path, test_emb_path, thresholds):
    db_embeddings = load_and_prepare_embeddings(os.path.join(db_emb_path, "database.pkl"))
    test_known_embeddings = load_and_prepare_embeddings(os.path.join(test_emb_path, "test_known.pkl"))
    test_unknown_embeddings = load_and_prepare_embeddings(os.path.join(test_emb_path, "test_unknown.pkl"))
    
    _, base_acc, _, _, _, _, _ = evaluate_accuracy(db_emb_path, test_emb_path, thresholds)
    max_deviation = 0
    distant_acc = 0

    for i in range(db_embeddings.shape[0]):
        mod_db_embeddings = np.delete(db_embeddings, i, axis=0)
        mod_known_embeddings = np.delete(test_known_embeddings, i, axis=0)
        
        known_sim_matrix = np.dot(mod_known_embeddings, mod_db_embeddings.transpose((0, 2, 1)))
        known_max_emb_value = np.max(known_sim_matrix, axis=-1)
        known_max_class_value = np.max(known_max_emb_value, axis=-1)
        known_max_class_index = np.argmax(known_max_emb_value, axis=-1)

        unknown_sim_matrix = np.dot(test_unknown_embeddings, mod_db_embeddings.transpose((0, 2, 1)))
        unknown_max_emb_value = np.max(unknown_sim_matrix, axis=-1)
        unknown_max_class_value = np.max(unknown_max_emb_value, axis=-1)

        fars, frrs, accuracies, misses = [], [], [], []
        total_known = mod_known_embeddings.shape[0] * mod_known_embeddings.shape[1]
        total_unknown = test_unknown_embeddings.shape[0] * test_unknown_embeddings.shape[1]
        for threshold in thresholds:
            mask = known_max_class_value > threshold
            detected = np.sum(np.sum(mask))
            indices_above_thresh = np.copy(known_max_class_index)
            indices_above_thresh[~mask] = -1
            labels = np.arange(indices_above_thresh.shape[0])[:, np.newaxis]
            correct_predictions = np.sum(indices_above_thresh == labels)
            false_rejects = indices_above_thresh.size - detected
            wrong_user = detected - correct_predictions

            mask = unknown_max_class_value > threshold
            false_accepts = np.sum(np.sum(mask))
            correct_predictions += mask.size - np.sum(np.sum(mask))

            accuracy = correct_predictions / (total_known + total_unknown)
            far = false_accepts / total_unknown
            frr = false_rejects / total_known
            miss = wrong_user / total_known

            accuracies.append(accuracy)
            fars.append(far)
            frrs.append(frr)
            misses.append(miss)

        abs_differences = np.abs(np.array(fars) - np.array(frrs))
        min_index = np.argmin(abs_differences)
        acc_at_err = accuracies[min_index]
        
        if (abs(acc_at_err - base_acc) > max_deviation):
            distant_acc = acc_at_err
            max_deviation = abs(acc_at_err - base_acc)
    return base_acc, distant_acc, max_deviation
    