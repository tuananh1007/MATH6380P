import os

from torchvision import datasets
from torchvision import transforms


class TransformedMNIST:
    def __init__(self):
        super(TransformedMNIST, self).__init__()
        # all the pretrained networks use the same normalization transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset_transforms = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ]
        )
        self.train_dataset = datasets.MNIST(os.getcwd() + "/mnist",
                                            train=True,
                                            transform=dataset_transforms,
                                            download=True)
        self.test_dataset = datasets.MNIST(os.getcwd() + "/mnist",
                                           train=False,
                                           transform=dataset_transforms,
                                           download=True)

    # now we will define specific functions for different pretrained convnets that we want to use

    def get_train(self):
        return (self.train_dataset)

    def get_test(self):
        return (self.test_dataset)
