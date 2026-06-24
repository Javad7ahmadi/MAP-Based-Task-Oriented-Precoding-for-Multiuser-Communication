import torch
import numpy as np
from scipy.io import savemat
from torch.utils.data import DataLoader

from Config import *
from datasets.multiview_cifar import CIFAR10MultiView
from models.multiview_encoder import MultiViewEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_data = CIFAR10MultiView(
    K=K,
    train=True,
    classes=chosen_classes
)

test_data = CIFAR10MultiView(
    K=K,
    train=False,
    classes=chosen_classes
)
#train_data = stratified_subset(train_data, num_per_class=399)
#test_data  = stratified_subset(test_data, num_per_class=90)


train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE_TEST,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


model = MultiViewEncoder(
    K=K,
    D_k=D_K,
    weights=WEIGHTS
).to(device)


if MODE == "MCR2":
    checkpoint_name = "MCR2_checkpoint.pth"

elif MODE == "OURLOSS":
    checkpoint_name = "OurLoss_checkpoint.pth"

checkpoint = torch.load(checkpoint_name)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

loss_list = checkpoint.get("loss_list", [])
ratio_list = checkpoint.get("ratio_list", [])


def extract_features(loader):
    all_Z = []
    all_y = []

    with torch.no_grad():
        for x_list, y in loader:
            x_list = [x.to(device) for x in x_list]

            Z, _ = model(x_list)

            Z_c = Z[:, 0::2] + 1j * Z[:, 1::2]

            norm = torch.sqrt(torch.sum(torch.abs(Z_c) ** 2, dim=1, keepdim=True))
            Z_c = Z_c / (norm + 1e-8)

            all_Z.append(Z_c.cpu())
            all_y.append(y)

    return torch.cat(all_Z, dim=0), torch.cat(all_y, dim=0)


Z_train, y_train = extract_features(train_loader)

mu = Z_train.mean(dim=0, keepdim=True)
Zc = Z_train - mu

C_xx = (Zc.conj().T @ Zc) / Z_train.shape[0]

CLS_mean = torch.stack(
    [Z_train[y_train == i].mean(dim=0) for i in range(num_classes )],
    dim=1
)

CLS_cov = []
CLS_rlt = []

for i in range(num_classes ):
    Zi = Z_train[y_train == i]
    mu_i = Zi.mean(dim=0, keepdim=True)
    Zi_c = Zi - mu_i

    CLS_cov.append((Zi_c.conj().T @ Zi_c) / Zi.shape[0])
    CLS_rlt.append((Zi_c.T @ Zi_c.conj()) / Zi.shape[0])

CLS_cov = torch.stack(CLS_cov, dim=2)
CLS_rlt = torch.stack(CLS_rlt, dim=2)

p = torch.bincount(y_train) / y_train.shape[0]

if MODE == "MCR2":
    statistics_name = "mcr2_statistics.mat"
    test_name = "mcr2_test_results.mat"

elif MODE == "OURLOSS":
    statistics_name = "OurStatistics.mat"
    test_name = "OurTest.mat"

savemat(statistics_name, {
    "CLS_mean": CLS_mean.cpu().numpy(),
    "CLS_cov": CLS_cov.cpu().numpy(),
    "CLS_rlt": CLS_rlt.cpu().numpy(),
    "C_xx": C_xx.cpu().numpy(),
    "Z": Z_train.cpu().numpy(),
    "labels": y_train.cpu().numpy(),
    "loss_curve": np.array(loss_list),
    "ratio_curve": np.array(ratio_list),
    "p": p.cpu().numpy()
})


Z_test, y_test = extract_features(test_loader)

savemat(test_name, {
    "test_feature_cplx": Z_test.cpu().numpy(),
    "test_label": y_test.cpu().numpy()
})

print("Export finished.")