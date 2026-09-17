import os
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import scipy.io
import random

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from models.MAP import loss as map_loss
from models.CenterLoss import CenterLoss
from models.DG import loss as dg_loss
from models.MCR2 import MaximalCodingRateReduction
from models.ContrastiveLoss import ContrastiveLoss
# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "./datasets/modelnet40_images_new_12x"
NUM_VIEWS = 2
D_FEATURE = 10
R = 16
BETA_MIN = 3e-4
BETA_MAX = 90
MAX_TRAIN_SAMPLES = 300
BATCH_SIZE = 64
#transforms.Resize((128,128))
NUM_WORKERS = 6

NUM_CLASSES = 10

CLASS_PRIORS = torch.tensor(
   # [0.40, 0.15, 0.10, 0.08, 0.07,
   #  0.06, 0.05, 0.04, 0.03, 0.02],
    [0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1],
    dtype=torch.float32
)
R_VALUES = [8, 16, 24, 32, 40, 48, 56, 64]

#BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20

#NUM_VIEWS = 2
REAL_FEATURE_DIM = 2 * D_FEATURE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mcr2_loss = MaximalCodingRateReduction(
    gam1=1.0,
    gam2=1.0,
    eps=0.01
).to(DEVICE)
#NUM_WORKERS = 4


# ============================================================
# Dataset
# ============================================================

class ModelNetMVCNN(Dataset):

    def __init__(self, root, split, class_to_idx=None, class_priors=None):

        self.root = root
        self.split = split

        self.samples = []
        classes = [
            #"bathtub",
            #"airplane",
            "bed",
            "chair",
            "desk",
            "guitar",
            "dresser",
            "monitor",
            #"night_stand",
            "sofa",
            "table",
            "xbox",
            "toilet"
        ]

        if class_to_idx is None:
            self.class_to_idx = {
                c:i for i,c in enumerate(classes)
            }
        else:
            self.class_to_idx = class_to_idx
            
        self.class_to_idx = {
            c:i for i,c in enumerate(classes)
        }


        for cls in classes:

            split_dir = os.path.join(
                root,
                cls,
                split
            )

            if not os.path.isdir(split_dir):
                continue


            objects = defaultdict(list)


            for f in os.listdir(split_dir):

                if not f.endswith(".png"):
                    continue


                # example:
                # laptop_0001.obj.shaded_v002.png

                obj_id = f.split(".obj")[0]

                objects[obj_id].append(
                    os.path.join(split_dir,f)
                )


            for obj, imgs in objects.items():

                if len(imgs) < NUM_VIEWS:
                    continue


                # sort views
                imgs = sorted(
                    imgs,
                    key=lambda x:
                    int(
                        re.search(
                            r"v(\d+)",
                            x
                        ).group(1)
                    )
                )


                self.samples.append(
                    (
                        imgs[:NUM_VIEWS],
                        self.class_to_idx[cls]
                    )
                )


        print(
            f"{split}: Loaded samples:",
            len(self.samples)
        )
        if split == "train" and MAX_TRAIN_SAMPLES is not None:

            random.seed(42)

            if len(self.samples) > MAX_TRAIN_SAMPLES:
                self.samples = random.sample(
                    self.samples,
                    MAX_TRAIN_SAMPLES
                )

            print(
                f"{split}: Using samples:",
                len(self.samples)
            )        

        if split == "test" and class_priors is not None:

            selected_samples = []

            for c in range(NUM_CLASSES):

                class_samples = [
                    sample for sample in self.samples
                    if sample[1] == c
                ]

                num_samples = int(
                    len(self.samples) * class_priors[c].item()
                )

                num_samples = min(
                    num_samples,
                    len(class_samples)
                )

                selected_samples.extend(
                    random.sample(
                        class_samples,
                        num_samples
                    )
                )

            self.samples = selected_samples

            print(
                f"{split}: Nonuniform samples:",
                len(self.samples)
            )



        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]
                )
            ]
        )


    def __len__(self):
        return len(self.samples)



    def __getitem__(self,index):

        paths,label = self.samples[index]

        views=[]


        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except:
                continue


            img = self.transform(img)

            views.append(img)


        if len(views) != NUM_VIEWS:
            return self.__getitem__((index + 1) % len(self.samples))

        views=torch.stack(views)

        # [12,3,224,224]

        return views,label



# ============================================================
# Network
# ============================================================

class MVCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.viewers = nn.ModuleList()


        for k in range(NUM_VIEWS):

            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )

            feature_extractor = nn.Sequential(
                *list(resnet.children())[:-1]
            )


            fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    512,
                    REAL_FEATURE_DIM
                ),
                #nn.ReLU()
            )


            self.viewers.append(
                nn.Sequential(
                    feature_extractor,
                    fc
                )
            )

            


    def forward(self,x):

        B,V,C,H,W = x.shape

        features=[]

        for k in range(NUM_VIEWS):

            xk = x[:,k,:,:,:]

            zk = self.viewers[k](xk)
            zero_mask = torch.norm(zk, dim=1) < 1e-10

            zc = complex_normalize(zk)


            features.append(zc)
        # Concatenate normalized complex features
        z = torch.cat(
            features,
            dim=1
        )

        return z
        
def map_classifier(
    features,
    class_means,
    class_covariances,
    priors,
    reg=1e-8
):
    """
    Exact MAP detector in the feature domain
    using full covariance matrices.

    features:
        [B, D] complex

    class_means:
        [C, D] complex

    class_covariances:
        [C, D, D] complex

    priors:
        [C]
    """

    B, D = features.shape
    C = class_means.shape[0]

    scores = []

    eye = torch.eye(
        D,
        device=features.device,
        dtype=features.dtype
    )

    for c in range(C):

        mu = class_means[c]

        # Regularized covariance
        Sigma = (
            class_covariances[c]
            + reg * eye
        )

        # Difference z - mu
        diff = (
            features - mu
        )

        # Solve:
        # Sigma^{-1} diff
        #
        # diff is [B,D], so transpose to [D,B]
        solved = torch.linalg.solve(
            Sigma,
            diff.T
        ).T

        # Mahalanobis distance:
        #
        # (z-mu)^H Sigma^{-1}(z-mu)
        mahalanobis = torch.sum(
            diff.conj() * solved,
            dim=1
        ).real

        # log |Sigma|
        sign, logdet = torch.linalg.slogdet(
            Sigma
        )

        logdet = logdet.real

        # MAP cost:
        #
        # Mahalanobis + log|Sigma| - log p_c
        cost = (
            mahalanobis
            + logdet
            - torch.log(
                priors[c] + 1e-12
            )
        )

        scores.append(cost)

    scores = torch.stack(
        scores,
        dim=1
    )

    predictions = scores.argmin(
        dim=1
    )

    return predictions
def get_class_statistics(model, loader):
    """
    Estimate full complex Gaussian statistics from all training samples.

    Returns:
        class_means      : [C, D] complex
        class_covariances: [C, D, D] complex Hermitian
    """

    model.eval()

    features_all = []
    labels_all = []

    with torch.no_grad():

        for views, y in loader:

            views = views.to(DEVICE)
            y = y.to(DEVICE)


            zc = model(views)

            features_all.append(zc)
            labels_all.append(y)
    features_all = torch.cat(
        features_all,
        dim=0
    )

    labels_all = torch.cat(
        labels_all,
        dim=0
    )

    class_means = []
    class_covariances = []

    D = features_all.shape[1]

    for c in range(NUM_CLASSES):

        class_features = features_all[
            labels_all == c
        ]

        # Mean
        mu = class_features.mean(dim=0)

        # Centered features
        centered = class_features - mu

        # Full complex covariance:
        #
        # Sigma = E[
        #   (z-mu)(z-mu)^H
        # ]
        covariance = (
            centered.conj().T @ centered
        ) / class_features.shape[0]

        class_means.append(mu)
        class_covariances.append(covariance)

    class_means = torch.stack(
        class_means
    )

    class_covariances = torch.stack(
        class_covariances
    )

    return class_means, class_covariances

def get_global_covariance(model, loader):
    """
    Estimate the covariance matrix of all training features.

    Returns:
        C_xx : [D, D] complex covariance matrix
    """

    model.eval()

    features_all = []

    with torch.no_grad():

        for views, y in loader:

            views = views.to(DEVICE)

            zc = model(views)

            features_all.append(zc)

    features_all = torch.cat(
        features_all,
        dim=0
    )

    # Global mean
    mean_all = features_all.mean(dim=0)

    # Centered features
    centered = features_all - mean_all

    # Complex covariance
    C_xx = (
        centered.conj().T @ centered
    ) / features_all.shape[0]

    return C_xx
# ============================================================
# Complex feature
# ============================================================

def complex_normalize(z):

    real = z[:,0::2]

    imag = z[:,1::2]


    zc=torch.complex(
        real.float(),
        imag.float()
    )


    power=torch.norm(
        zc,
        dim=1,
        keepdim=True
    )


    zc=zc/(power+1e-8)


    return zc
def minimum_distance_classifier(features, class_means):
    """
    Minimum-distance classifier.

    features:    [B, D] complex
    class_means: [C, D] complex
    """

    distances = torch.abs(
        features.unsqueeze(1) -
        class_means.unsqueeze(0)
    ) ** 2

    distances = distances.sum(dim=2)

    predictions = distances.argmin(dim=1)

    return predictions
def get_class_means(model, loader):

    model.eval()

    features_all = []
    labels_all = []

    with torch.no_grad():

        for views, y in loader:

            views = views.to(DEVICE)
            y = y.to(DEVICE)

            zc = model(views)

            features_all.append(zc)
            labels_all.append(y)

    features_all = torch.cat(features_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    class_means = []

    for c in range(NUM_CLASSES):

        class_features = features_all[
            labels_all == c
        ]

        class_mean = class_features.mean(dim=0)

        class_means.append(class_mean)

    class_means = torch.stack(class_means)

    return class_means
# ============================================================
# Ratio metric
# ============================================================

def correlation_ratio(features,labels):


    classes=torch.unique(labels)


    means=[]

    intra=[]


    for c in classes:

        x=features[
            labels==c
        ]


        if len(x)<2:
            continue


        mean=x.mean(dim=0)

        means.append(mean)


        d=torch.mean(
            torch.abs(
                x-mean
            )**2
        )

        intra.append(d)



    if len(means)<2:
        return torch.tensor(0.0,device=features.device)


    intra=torch.stack(intra).mean()


    inter=[]


    for i in range(len(means)):

        for j in range(i+1,len(means)):

            d=torch.abs(
                means[i]-means[j]
            )**2

            inter.append(
                torch.mean(d)
            )


    inter=torch.stack(inter).mean()


    ratio=inter/(intra+1e-8)


    return ratio



# ============================================================
# Training
# ============================================================
def select_loss():

    print("Select loss function:")
    print("1. MAP")
    print("2. MCR2")
    print("3. Center Loss")
    print("4. Contrastive Loss")
    print("5. Discriminative Gain")

    choice = int(input("Enter choice (1-5): "))

    if choice == 1:
        return map_loss, "MAP"

    elif choice == 3:
        return "CenterLoss", "CenterLoss"

    elif choice == 4:
        return "ContrastiveLoss", "ContrastiveLoss"

    elif choice == 2:
        return mcr2_loss, "MCR2"
    elif choice == 5:
        return dg_loss, "DG"

    else:
        raise ValueError(
            "Please enter a number between 1 and 5."
        )

def train():
    loss_function, loss_name = select_loss()
    train_set=ModelNetMVCNN(
        DATA_ROOT,
        "train"
    )

    test_set=ModelNetMVCNN(
        DATA_ROOT,
        "test",
        train_set.class_to_idx,
        CLASS_PRIORS
    )
    train_loader=DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader=DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )




    model=MVCNN().to(DEVICE)
    
    with torch.no_grad():
        dummy = torch.zeros(
            1, NUM_VIEWS, 3, 224, 224,
            device=DEVICE
        )
        feature_dim = model(dummy).shape[1]

    print("Actual feature dimension:", feature_dim)
    if loss_name == "CenterLoss":

        center_loss = CenterLoss(
            NUM_CLASSES,
            feature_dim,
            DEVICE
        )
    if loss_name == "ContrastiveLoss":

        contrastive_loss = ContrastiveLoss(
            margin=1.0
        ).to(DEVICE)        
    if loss_name == "CenterLoss":

        optimizer = optim.Adam(
            list(model.parameters()) +
            list(center_loss.parameters()),
            lr=LR,
            weight_decay=1e-5
        )

    else:

        optimizer = optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=1e-5
        )

    scaler = torch.cuda.amp.GradScaler()

    test_accuracies = []

    test_feature_cplx2 = []
    test_label2 = []

    CLS_mean = []
    CLS_cov = []
    C_xx = []
    for epoch in range(EPOCHS):
        model.train()

        total_loss=0

        total_correct=0

        total=0


        start=time.time()


        for step,(views,y) in enumerate(train_loader):


            views=views.to(DEVICE)

            y=y.to(DEVICE)


            optimizer.zero_grad()


            zc=model(views)



            if loss_name == "MAP":

                R = random.choice(R_VALUES)

                loss = loss_function(
                    zc,
                    y,
                    CLASS_PRIORS.to(DEVICE),
                    BETA_MIN,
                    BETA_MAX
                )

            elif loss_name == "CenterLoss":

                loss = center_loss(
                    zc,
                    y
                )
            elif loss_name == "ContrastiveLoss":

                loss = contrastive_loss(
                    zc,
                    y
                )

            elif loss_name == "MCR2":

                loss = loss_function(
                    zc,
                    y,
                )

            else:

                loss = loss_function(
                    zc,
                    y
                )

                #loss = loss_function(logits, y)

            if loss_name == "CenterLoss":
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    
            




            total_loss+=loss.item()






            total+=len(y)



        # Calculate class means using the trained model
        class_means, class_covariances = get_class_statistics(
            model,
            train_loader
        )
        CLS_mean.append(
            class_means.detach().cpu().numpy().T
        )


        CLS_cov.append(
            class_covariances.detach().cpu().numpy().transpose(1, 2, 0)
        )
        # Calculate covariance of all training features
        global_covariance = get_global_covariance(
            model,
            train_loader
        )

        C_xx.append(
            global_covariance.detach().cpu().numpy()
        )        
        # Calculate training accuracy using minimum-distance classifier
        model.eval()

        total_correct = 0
        total = 0

        with torch.no_grad():

            for views, y in train_loader:

                views = views.to(DEVICE)
                y = y.to(DEVICE)

                zc = model(views)

                pred = map_classifier(
                    zc,
                    class_means,
                    class_covariances,
                    CLASS_PRIORS
                )
                total_correct += (
                    pred == y
                ).sum().item()

                total += len(y)

        train_acc = 100 * total_correct / total



        # ============================================================
        # Test
        # ============================================================

        model.eval()

        correct = 0
        total = 0
        ratios = []

        # Store all test features and labels for this epoch
        epoch_test_features = []
        epoch_test_labels = []

        with torch.no_grad():

            for views, y in test_loader:

                views = views.to(DEVICE)
                y = y.to(DEVICE)

                zc = model(views)

                # Save test features
                epoch_test_features.append(
                    zc.cpu()
                )

                # Save corresponding labels
                epoch_test_labels.append(
                    y.cpu()
                )

                # Exact MAP detector
                pred = map_classifier(
                    zc,
                    class_means,
                    class_covariances,
                    CLASS_PRIORS.to(DEVICE)
                )

                correct += (
                    pred == y
                ).sum().item()

                total += len(y)

                ratios.append(
                    correlation_ratio(
                        zc,
                        y
                    )
                )


        # ------------------------------------------------------------
        # Concatenate all test batches
        # ------------------------------------------------------------

        epoch_test_features = torch.cat(
            epoch_test_features,
            dim=0
        )

        epoch_test_labels = torch.cat(
            epoch_test_labels,
            dim=0
        )


        # ------------------------------------------------------------
        # Save this epoch
        # ------------------------------------------------------------

        test_feature_cplx2.append(
            epoch_test_features.numpy()
        )

        test_label2.append(
            epoch_test_labels.numpy()
        )




        test_acc=100*correct/total
        test_accuracies.append(test_acc)

        ratio=torch.stack(ratios).mean()



        print(
            f"Epoch {epoch} | "
            f"Train MAP Error = {total_loss/len(train_loader):.8f} | "
            f"Test Acc {test_acc:.2f}% | "
            f"Ratio {ratio.item():.4f} | "
            f"Time {time.time()-start:.1f}s"
        )

    # Convert to row vector
    test_accuracies = torch.tensor(test_accuracies).numpy().reshape(1, -1)

    # Print accuracy vector
    print("\nTest Accuracy Vector:")
    print(test_accuracies)

    


    base_filename = f"saved_MAT"

    counter = 0
    save_filename = f"{base_filename}.mat"

    while os.path.exists(save_filename):
        counter += 1
        save_filename = f"{base_filename}.mat"

    scipy.io.savemat(
        save_filename,
        {
            "test_accuracy": test_accuracies,
            "test_feature_cplx2": test_feature_cplx2,
            "test_label2": test_label2,
            "CLS_mean": CLS_mean,
            "CLS_cov": CLS_cov,
            "C_xx": C_xx,
            "K": NUM_VIEWS,
            "D_k": D_FEATURE,
            "R_VALUES": R_VALUES,
            "class_priors": CLASS_PRIORS.numpy(),
            "loss_name": loss_name
        }
    )

    print(f"\nSaved to {save_filename}")



if __name__=="__main__":

    train()