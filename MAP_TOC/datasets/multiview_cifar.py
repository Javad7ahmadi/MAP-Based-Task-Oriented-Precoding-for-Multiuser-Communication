import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import numpy as np
import torchvision.transforms as T

class CIFAR10MultiView:

    def __init__(self, root="./data", train=True, K=2, classes=None):

        self.K = K

        dataset = datasets.CIFAR100(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
        self.aug = T.Compose([
            T.RandomResizedCrop(32, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor()
        ])


        # =========================
        # choose only selected classes
        # =========================
        if classes is not None:

            targets = np.array(dataset.targets)

            class_to_new = {
                c: i for i, c in enumerate(classes)
            }

            indices = []
            new_labels = []

            for i, t in enumerate(targets):

                if t in class_to_new:

                    indices.append(i)
                    new_labels.append(class_to_new[t])

            self.indices = indices
            self.labels = new_labels
            self.dataset = dataset

        else:

            self.dataset = dataset
            self.indices = None
            self.labels = None

    def __len__(self):

        if self.indices is not None:
            return len(self.indices)

        return len(self.dataset)

    #def make_views(self, img):

    #    views = []

    #    # img is already [C, H, W] tensor

    #    C, H, W = img.shape

    #    slice_height = H // self.K

    #    for k in range(self.K):

    #        start = k * slice_height

    #        if k == self.K - 1:
    #            end = H
    #        else:
    #            end = (k + 1) * slice_height

    #        view = img[:, start:end, :]

    #        views.append(view)

    #    return views    
    def make_views(self, img):

        views = []

        img = TF.to_pil_image(img)

        for _ in range(self.K):
            views.append(self.aug(img))

        return views

    # def make_views(self, img):

    #     views = []

    #     for k in range(self.K):

    #         angle = k * (360 / self.K)

    #         rotated = TF.rotate(img, angle)

    #         views.append(rotated)

    #     return views

    def __getitem__(self, idx):

        if self.indices is not None:

            real_idx = self.indices[idx]

            img, _ = self.dataset[real_idx]

            label = self.labels[idx]

        else:

            img, label = self.dataset[idx]

        views = self.make_views(img)

        return views, label