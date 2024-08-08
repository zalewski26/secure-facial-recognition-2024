import argparse
import os
import pickle
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from deepface.basemodels import ArcFace, Facenet, GhostFaceNet

from utils.experiment_parameters import ExperimentParameters
from utils.utils import Model, Modification, get_dataset_path, get_embedding_path


def _preprocess_image(image_path: str, input_shape: tuple = (160, 160)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, input_shape)
    img = img.astype("float32")
    img /= 255.0
    return img


def preprocess_all_images(images_path, model_shape):
    return np.array(
        [
            _preprocess_image(os.path.join(os.path.join(images_path, class_name), image_name), model_shape)
            for class_name in os.listdir(images_path)
            for image_name in os.listdir(os.path.join(images_path, class_name))
        ]
    )


def process_images_in_batch(model, image_path, model_shape, batch_size=128):
    result = []
    images = preprocess_all_images(image_path, model_shape)
    for i in range(len(images) // batch_size + 1):
        temp_result = model(images[i * batch_size : min((i + 1) * batch_size, len(images))])
        result.extend(temp_result)
    return np.array(result)


def get_model(model_name):
    if model_name == "Facenet":
        model = Facenet.load_facenet128d_model()
        model_shape = (160, 160)
    elif model_name == "ArcFace":
        model = ArcFace.load_model()
        model_shape = (112, 112)
    elif model_name == "GhostFaceNet":
        model = GhostFaceNet.load_model()
        model_shape = (112, 112)
    return model, model_shape


def create_embeddings(num_classes, num_of_db_images, num_of_test_images, model_name, modification, mod_param):
    start = time.time()
    dataset_path = get_dataset_path(num_classes, num_of_db_images, num_of_test_images, modification, mod_param)

    db_embeddings = {}
    test_known_embeddings = {}
    test_unknown_embeddings = {}

    db_path = os.path.join(dataset_path, "database")
    test_known_path = os.path.join(dataset_path, "test_known")
    test_unknown_path = os.path.join(dataset_path, "test_unknown")

    model, model_shape = get_model(model_name)

    db_emb_list = process_images_in_batch(model, db_path, model_shape)
    known_emb_list = process_images_in_batch(model, test_known_path, model_shape)
    unknown_emb_list = process_images_in_batch(model, test_unknown_path, model_shape)

    db_embeddings = {
        class_name: db_emb_list[i * num_of_db_images : (i + 1) * num_of_db_images]
        for i, class_name in enumerate(os.listdir(db_path))
    }
    test_known_embeddings = {
        class_name: known_emb_list[i * num_of_test_images : (i + 1) * num_of_test_images]
        for i, class_name in enumerate(os.listdir(test_known_path))
    }
    test_unknown_embeddings = {
        class_name: unknown_emb_list[i * num_of_test_images : (i + 1) * num_of_test_images]
        for i, class_name in enumerate(os.listdir(test_unknown_path))
    }

    embedding_path = get_embedding_path(
        num_classes, num_of_db_images, num_of_test_images, model_name, modification, mod_param
    )
    os.makedirs(embedding_path, exist_ok=True)
    with open(f"{embedding_path}/database.pkl", "wb") as f:
        pickle.dump(db_embeddings, f)
    with open(f"{embedding_path}/test_known.pkl", "wb") as f:
        pickle.dump(test_known_embeddings, f)
    with open(f"{embedding_path}/test_unknown.pkl", "wb") as f:
        pickle.dump(test_unknown_embeddings, f)

    end = time.time()
    print(
        f"Completed: {num_classes}_{num_of_db_images}_{num_of_test_images}_{model_name}"
        + f" ({modification or 'clean'}) ({(end - start):.3f} s)"
    )


def create_all_embeddings():
    for num_classes in ExperimentParameters.CLASSES_NUMBER:
        for num_of_db_images in ExperimentParameters.DB_IMAGES:
            for num_of_test_images in ExperimentParameters.TEST_IMAGES:
                for model in ExperimentParameters.MODELS:
                    model_name = str(model)
                    for modification in ExperimentParameters.MODIFICATIONS:
                        if modification is None:
                            mod_params = [None]
                        else:
                            mod_params = ExperimentParameters.MOD_PARAMS[modification]
                        for mod_param in mod_params:
                            create_embeddings(
                                num_classes, num_of_db_images, num_of_test_images, model_name, modification, mod_param
                            )


def main():
    parser = argparse.ArgumentParser(description="Create embeddings for a given dataset.")
    parser.add_argument(
        "--mode", type=str, choices=["all", "one"], default="one", help="Run mode: all configurations or one."
    )
    parser.add_argument("--num_classes", type=int, help="Number of classes.")
    parser.add_argument("--database_images", type=int, help="Number of images in database per class.")
    parser.add_argument("--test_images", type=int, help="Number of test images per class.")
    parser.add_argument("--model", type=str, choices=[m.name for m in Model], help="Model name.")
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
        create_all_embeddings()
    else:
        model_name = str(Model[args.model])
        modification = Modification[args.modification] if args.modification else None
        create_embeddings(
            args.num_classes, args.database_images, args.test_images, model_name, modification, args.mod_param
        )


if __name__ == "__main__":
    main()
