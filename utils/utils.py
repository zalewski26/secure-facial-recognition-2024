from enum import Enum, auto


class Model(Enum):
    FACENET = auto()
    ARCFACE = auto()
    GHOSTFACENET = auto()

    def __str__(self):
        if self == Model.FACENET:
            return "Facenet"
        elif self == Model.ARCFACE:
            return "ArcFace"
        elif self == Model.GHOSTFACENET:
            return "GhostFaceNet"


class Modification(Enum):
    BLUR = auto()
    ELASTIC = auto()
    PERMUTE = auto()
    FAWKES = auto()
    LOWKEY = auto()
    STYLE = auto()

    def __str__(self):
        return self.name.lower()


def get_dataset_path(num_classes, train_images, test_images, modification=None, mod_param=None):
    dataset_path = f"datasets/subsets_facescrub/{num_classes}classes_{train_images}train_{test_images}test"
    if modification is None:
        dataset_path = f"{dataset_path}/clean"
    else:
        dataset_path = f"{dataset_path}/{str(modification)}_{mod_param}"
    return dataset_path


def get_embedding_path(num_classes, train_images, test_images, model_name, modification=None, mod_param=None):
    embedding_path = f"embeddings/{num_classes}_{train_images}_{test_images}_{model_name}"
    if modification is None:
        embedding_path = f"{embedding_path}/clean"
    else:
        embedding_path = f"{embedding_path}/{str(modification)}_{mod_param}"
    return embedding_path
