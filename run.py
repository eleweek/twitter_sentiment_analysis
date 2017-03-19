import sys
import models
import datasets
import inspect
import logging

logging.basicConfig(level=logging.INFO)


def find_model_class_by_name(model_class_name):
    model = None
    for name, obj in inspect.getmembers(models):
        if inspect.isclass(obj) and name == model_class_name or getattr(obj, "model_name", None) == model_class_name:
            model = obj

    return model


def train_features_model(args):
    dataset_name = args[0]
    model_class_name = args[1]
    model_file_name = args[2]

    model_class = find_model_class_by_name(model_class_name)
    if model_class is None:
        raise Exception("Unknown model name {}".format(model_class_name))

    load_dataset_from_directory = getattr(datasets, "load_{}_from_directory".format(dataset_name))
    dataset_dir = "datasets/{}".format(dataset_name)
    if load_dataset_from_directory is None:
        raise Exception("Unknown dataset name {}".format(dataset_name))

    model = model_class.create_from_argv(*args[3:])
    dataset_train, dataset_test = load_dataset_from_directory(dataset_dir)

    model.train(dataset_train)
    model.save(model_file_name)


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "train_features_model":
        train_features_model(sys.argv[2:])
    else:
        raise Exception("Unknown action {}".format(action))
