import os
from time import time

import torch
from torch.utils.data.dataset import TensorDataset

"""generic class to extract features from pretrained nets"""


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        try:
            self.model.eval()
        except AttributeError:
            if self.model.__class__.__name__ in "Scattering2D":
                pass
            else:
                raise

        try:
            for param in self.model.parameters():
                param.requires_grad: bool = False
        except AttributeError:
            if self.model.__class__.__name__ in "Scattering2D":
                pass
            else:
                raise

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def features(self, dataloader, save_to_disk=True, train=True, flatten_config=None):
        feat_coll = []
        label_coll = []

        # only need to do this once before the loop starts
        # should I move the model back to CPU after this?
        try:
            self.model = self.model.to(self.device)
        except AttributeError:
            pass

        for batch_id, [features, labels] in enumerate(dataloader):
            # sample is a list with the first element corresponding to the images
            print("Batch {}, features shape: {}, labels shape: {}".format(batch_id, features.shape, labels.shape))
            features = features.to(self.device)
            labels = labels.to(self.device)

            t1 = time()
            out = self.model(features)
            t2 = time()
            print("Output shape: {}, Time taken: {}".format(out.shape, t2 - t1))

            # move output, features and labels back to the CPU to prevent a memory leak and release memory from GPU
            out = out.to("cpu")
            features = features.to("cpu")
            # do not need to move labels to GPU because we are not doing any computation on them
            # labels = labels.to("cpu")

            if flatten_config is not None:
                try:
                    start_dim = flatten_config["start_dim"]
                except KeyError:
                    start_dim = 0

                try:
                    end_dim = flatten_config["end_dim"]
                except KeyError:
                    end_dim = -1

                out = torch.flatten(out, start_dim=start_dim, end_dim=end_dim)
                print("Flattend output shape: {}".format(out.shape))

            feat_coll.append(out)
            label_coll.append(labels)

        out_features = torch.flatten(torch.stack(feat_coll), start_dim=0, end_dim=1)
        out_labels = torch.flatten(torch.stack(label_coll), start_dim=0, end_dim=1)

        print("The final features matrix has shape: {}".format(out_features.shape))

        if save_to_disk:
            # save as TensorDataset
            out_dataset = TensorDataset(out_features, out_labels)
            if train:
                prefix = "train"
            else:
                prefix = "test"
            filename = "{}_{}_dataset.pt".format(prefix, self.model.__class__.__name__)
            torch.save(out_dataset, filename)
            print("Saved features at {}/{}".format(os.getcwd(), filename))

        return out_features, out_labels