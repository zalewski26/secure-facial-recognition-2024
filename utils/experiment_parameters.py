from utils.utils import Model, Modification


class ExperimentParameters:
    CLASSES_NUMBER = [10, 50, 100]
    DB_IMAGES = [10]
    TEST_IMAGES = [30]
    MODELS = [Model.FACENET, Model.ARCFACE, Model.GHOSTFACENET]
    MODIFICATIONS = [
        None,
        Modification.BLUR,
        Modification.PERMUTE,
        Modification.FAWKES,
        Modification.LOWKEY,
        Modification.STYLE,
    ]
    MOD_PARAMS = {
        Modification.BLUR: [15],
        Modification.PERMUTE: [32],
        Modification.FAWKES: ["low", "mid", "high"],
        Modification.LOWKEY: [None],
        Modification.STYLE: ["vangogh"],
    }
