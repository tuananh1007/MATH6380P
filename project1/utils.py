import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import dump, load
from kymatio import Scattering2D
from matplotlib import cm
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from project1.ConvNetMods import alexnetmod, vgg16mod, resnetmod
from project1.Dataset import TransformedMNIST
from project1.FeatureExtractor import FeatureExtractor

if torch.cuda.is_available():
    pass
else:
    pass


def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30.0, n_iter=50000, init="pca")
    plot_only = 500
    embeddings = tsne.fit_transform(features.numpy()[:plot_only, :])
    labels = labels[:plot_only]
    plot_with_labels(embeddings, labels)


# visualize some features using tsne
def plot_with_labels(weights, labels):
    plt.cla()
    if type(labels[0]) == torch.Tensor:
        labels_plt = [x.item() for x in labels]
    else:
        labels_plt = labels

    X, Y = weights[:, 0], weights[:, 1]
    for x, y, s in zip(X, Y, labels_plt):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)

    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("Visualize features")


def imshow(inp, title=None, normalize=True):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean

    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, interpolation="bilinear", aspect="auto")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def extract_mnist_features(ignore=[], save_to_disk=True, train=True):
    models = []
    if "alexnet" not in ignore:
        alexnet = alexnetmod()
        models.append(alexnet)

    if "vgg16" not in ignore:
        vgg16 = vgg16mod()
        models.append(vgg16)

    if "resnet" not in ignore:
        resnet = resnetmod()
        models.append(resnet)
    # get dataset
    mnist = TransformedMNIST()
    batch_size = {"alexnetmod": 1000, "vgg16mod": 100, "resnetmod": 100}
    for model in models:
        name = model.__class__.__name__
        extractor = FeatureExtractor(model)
        dataset = mnist.get_train() if train else mnist.get_test()
        dataloader = DataLoader(dataset, batch_size=batch_size[name], num_workers=2)

        _ = extractor.features(dataloader, save_to_disk=save_to_disk, train=train)


def scattering_transform_mnist(save_to_disk=True, train=True):
    # here we want untransformed mnist data
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(os.getcwd() + "/mnist",
                                 train=True,
                                 transform=transform,
                                 download=True)
    mnist_test = datasets.MNIST(os.getcwd() + "/mnist",
                                train=False,
                                transform=transform,
                                download=True)

    # construct the scattering object
    scattering = Scattering2D(J=2, shape=(28, 28))
    batch_size = 1000 if torch.cuda.is_available() else 100
    dataloader = DataLoader(mnist_train if train else mnist_test, batch_size=batch_size)

    print("Running scattering transform")
    extractor = FeatureExtractor(scattering)
    out_features, out_labels = extractor.features(dataloader,
                                                  save_to_disk=save_to_disk,
                                                  train=train,
                                                  flatten_config={"start_dim": 2})


def get_dataset_dir(use_default=True):
    if use_default:
        print("reading from local directory")
        dataset_dir = "/home/sutd/Documents/Workspace/MATH6380P/project1/"
    else:
        print("not reading from local directory, probably need to save to new directory")
        dataset_dir = "~/math6380/project1/"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

    return dataset_dir


def train_datasets_filenames():
    dataset_dir = get_dataset_dir()

    return {
        "alexnet": dataset_dir + "train_alexnetmod_dataset.pt",
        "vgg16": dataset_dir + "train_vgg16mod_dataset.pt",
        "resnet": dataset_dir + "train_resnetmod_dataset.pt",
        "scattering": dataset_dir + "train_Scattering2D_dataset.pt"
    }


def get_train_dataset_filename(dataset_name):
    all_filenames = train_datasets_filenames()

    return all_filenames[dataset_name]


def test_datasets_filenames():
    dataset_dir = get_dataset_dir()

    return {
        "alexnet": dataset_dir + "test_alexnetmod_dataset.pt",
        "vgg16": dataset_dir + "test_vgg16mod_dataset.pt",
        "resnet": dataset_dir + "test_resnetmod_dataset.pt",
        "scattering": dataset_dir + "test_Scattering2D_dataset.pt"
    }


def get_test_dataset_filename(dataset_name):
    all_filenames = test_datasets_filenames()

    return all_filenames[dataset_name]


def get_stored_dataset(dataset, train=True):
    datasets = train_datasets_filenames() if train else test_datasets_filenames()
    to_load = datasets[dataset]
    loaded_dataset = torch.load(to_load)
    features = loaded_dataset[:][0]
    labels = loaded_dataset[:][1]

    if dataset == "scattering":
        features = torch.flatten(features, start_dim=1)

    return features, labels


def get_stored_model(model_name, dataset, cv=False):
    dataset_dir = get_dataset_dir()
    filename = "{}_{}_{}.joblib".format(model_name, "cv" if cv else "nocv", dataset)
    classifier = load(dataset_dir + filename)

    return classifier

# which sklearn models shoulud I train on?
# 1. Logistic regression
# 2. Random Forests
# 3. Gradient Boosting Trees
def train_classifier(classifier_to_train, dataset="alexnet", cv=False, save_to_disk=True):
    # build a pipeline wih the estimators that we need
    # first we need to scale and center the data
    log_reg = SGDClassifier(
        loss="log",
        penalty="elasticnet",
        tol=0.1,
        max_iter=1000,
        eta0=0.1,
        verbose=5
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        verbose=5
    )

    gbt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        verbose=5
    )

    classifiers = {"log_reg": log_reg, "random_forest": rf, "grad_boost": gbt}

    # param grid for cross validation
    param_grid = dict()
    param_grid["log_reg"] = {"learning_rate": ["optimal", "invscaling"]}
    param_grid["random_forest"] = {"n_estimators": [10, 100, 500]}
    param_grid["grad_boost"] = {
        "n_estimators": [10, 100, 500],
        "learning_rate": [0.1, 1],
        "subsample": [1.0, 0.5]
    }

    #  declare the standard scaler
    scaler = preprocessing.StandardScaler()

    classifier = classifiers[classifier_to_train]
    train_features, train_labels = get_stored_dataset(dataset, train=True)
    train_features = train_features.numpy()
    train_labels = train_labels.numpy()
    scaler.fit(train_features)

    classifier_to_fit = classifier if not cv else GridSearchCV(classifier, param_grid[classifier_to_train], verbose=5)

    classifier_to_fit.fit(scaler.transform(train_features), train_labels)

    if save_to_disk:
        filename = "{}_{}_{}.joblib".format(classifier_to_train, "cv" if cv else "nocv", dataset)
        dump(classifier_to_fit, get_dataset_dir() + filename)

    return classifier_to_fit


def test_classifier(classifier_to_test, dataset=["alexnet"], cv=False, load_from_disk=True):
    dataset_dir = get_dataset_dir()
    filename = "{}_{}_{}.joblib".format(classifier_to_test, "cv" if cv else "nocv", dataset)
    classifier = load(dataset_dir + filename)

    # stop doing this and change to use pipeline later!
    train_features, _ = get_stored_dataset(dataset, train=True)
    train_features = train_features.numpy()
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_features)

    test_features, _ = get_stored_dataset(dataset, train=False)
    test_features = test_features.numpy()

    preds = classifier.predict(scaler.transform(test_features))

    return preds
