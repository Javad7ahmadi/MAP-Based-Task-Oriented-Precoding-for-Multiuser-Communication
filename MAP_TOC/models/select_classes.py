import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configMCR2 import *
from datasets.multiview_cifar import CIFAR10MultiView
from models.multiview_encoder import MultiViewEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. load ALL CIFAR100 classes
train_data = CIFAR10MultiView(
    K=K,
    train=True,
    classes=None
)

train_loader = DataLoader(
    train_data,
    batch_size=256,
    shuffle=False,
    num_workers=4
)


# 2. load model
model = MultiViewEncoder(
    K=K,
    D_k=D_K,
    weights=WEIGHTS
).to(device)

checkpoint = torch.load("MCR2_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# 3. feature extraction
def extract_features(loader):
    all_Z = []
    all_y = []

    with torch.no_grad():
        for x_list, y in loader:

            x_list = [x.to(device) for x in x_list]

            Z, _ = model(x_list)

            Z = F.normalize(Z, dim=1)

            all_Z.append(Z.cpu())
            all_y.append(y)

    return torch.cat(all_Z), torch.cat(all_y)


# 4. run extraction
Z, y = extract_features(train_loader)

num_classes = 100

means = []

for c in range(num_classes):
    Zc = Z[y == c]
    means.append(Zc.mean(dim=0))

means = torch.stack(means)

means = F.normalize(means, dim=1)

S = means @ means.T

score = S.sum(dim=1) - torch.diag(S)

best_classes = torch.argsort(score)[:10]

print("\nBEST 10 CLASSES:\n")
print(best_classes.tolist())