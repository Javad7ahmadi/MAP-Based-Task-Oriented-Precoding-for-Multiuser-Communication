import torch
torch.backends.cudnn.benchmark = True
import time
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
from Config import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/mcr2_experiment")

from datasets.multiview_cifar import CIFAR10MultiView
from models.multiview_encoder import MultiViewEncoder
from models.mcr2_loss import MaximalCodingRateReduction
from models.our_loss import OurLoss
def extract_features(loader):
    all_Z = []
    all_y = []

    with torch.no_grad():
        for x_list, y in loader:
            x_list = [x.to(device) for x in x_list]

            Z, _ = model(x_list)

            D = Z.shape[1] // 2
            #Z_c = Z[:, :D] + 1j * Z[:, D:]
            Z_c = Z[:, 0::2] + 1j * Z[:, 1::2]
            norm = torch.sqrt(torch.sum(torch.abs(Z_c)**2, dim=1, keepdim=True))
            Z_c = Z_c / (norm + 1e-8)

            all_Z.append(Z_c.cpu())
            all_y.append(y)

    Z_c = torch.cat(all_Z, dim=0)
    y = torch.cat(all_y, dim=0)

    return Z_c, y




# device
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# dataset
from torch.utils.data import Subset
import numpy as np

def stratified_subset(dataset, num_per_class):
    targets = np.array([dataset[i][1] for i in range(len(dataset))])

    indices = []

    for c in np.unique(targets):
        class_idx = np.where(targets == c)[0]
        np.random.shuffle(class_idx)
        indices.extend(class_idx[:num_per_class])

    return Subset(dataset, indices)


# dataset

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

#train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
#test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
test_loader  = DataLoader(
    test_data,
    batch_size=BATCH_SIZE_TEST ,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# model
model = MultiViewEncoder(
    K=K,
    D_k=D_K,
    weights=WEIGHTS
).to(device)
model.train()


# optimizers
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
#optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-3)


# loss
if MODE == "MCR2":
    criterion = MaximalCodingRateReduction(
        gam1=1,
        gam2=1,
        eps=epsilon_MCR2
    )

elif MODE == "OURLOSS":
    criterion = OurLoss()

def compute_correlation_ratio(Z_c, y):
    num_classes = len(chosen_classes)
    intra_vals = []
    inter_vals = []

    # class means
    means = []

    for j in range(num_classes):
        Zj = Z_c[y == j]
        mu = Zj.mean(dim=0)
        means.append(mu)

        # intra-class correlation
        corr = torch.abs((Zj @ mu.conj()) /
                         (torch.norm(Zj, dim=1) * torch.norm(mu) + 1e-8))

        intra_vals.append(corr.mean())

    means = torch.stack(means)

    # inter-class correlation
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                corr = torch.abs(
                    torch.dot(means[i], means[j].conj())
                ) / (torch.norm(means[i]) * torch.norm(means[j]) + 1e-8)

                inter_vals.append(corr)

    intra = torch.stack(intra_vals).mean()
    inter = torch.stack(inter_vals).mean()

    ratio = intra / (inter + 1e-8)

    return ratio.item()

global_step = 0

loss_list = []
ratio_list = []
# training loop
for epoch in range(500):
    start_time = time.time()
    epoch_loss = 0
    n_batches = 0
    max_steps=10000000000000000 #################################################################################################
    for step, (x_list, y) in enumerate(train_loader):

        if step >= max_steps:
            break
        x_list = [x.to(device) for x in x_list]
        y = y.to(device)

        optimizer.zero_grad()
        #optimizer_cls.zero_grad()

        # forward
        Z, z_list = model(x_list)

        D = Z.shape[1] // 2
        #Z_c = Z[:, :D] + 1j * Z[:, D:]
        #Z_c = Z
        Z_c = Z[:, 0::2] + 1j * Z[:, 1::2]

        norm = torch.sqrt(torch.sum(torch.abs(Z_c)**2, dim=1, keepdim=True))
        Z_c = Z_c / (norm + 1e-8)

        #Z = torch.cat([Z_c.real, Z_c.imag], dim=1)
        Z = Z_c

        with torch.no_grad():
            ratio = compute_correlation_ratio(Z_c, y)
        

        #with torch.no_grad():
        #   logits = classifier(feat)
        #   pred = torch.argmax(logits, dim=1)
        #   acc = (pred == y).float().mean().item()


        # MCR2 loss
        loss, parts = criterion(Z, y)

        #print(torch.min(y), torch.max(y), torch.unique(y))
        loss_total = loss 

        # backprop
        loss_total.backward()
        optimizer.step()
        #optimizer_cls.step()

        # accuracy
        # pred = torch.argmax(logits, dim=1)
        # acc = (pred == y).float().mean()

        # accumulate epoch stats
        epoch_loss += loss_total.item()
        #epoch_acc += acc.item()
        n_batches += 1

        # ?? TensorBoard logging (real-time)

        global_step += 1

        if step % 10 == 0:
            print(
                f"Epoch {epoch} Step {step} "
                f"Loss {loss.item():.4f}, "
                f"Ratio {ratio:.4f}"
            )
        loss_list.append(loss.item())
        ratio_list.append(ratio)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} time: {epoch_time:.2f} sec")

    if epoch % 10 == 0:
        if MODE == "MCR2":
            checkpoint_name = "MCR2_checkpoint.pth"

        elif MODE == "OURLOSS":
            checkpoint_name = "OurLoss_checkpoint.pth"

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss_list": loss_list,
            "ratio_list": ratio_list
        }, checkpoint_name)        
        print("Saved checkpoint at epoch", epoch)    
    
    # ?? epoch-level logging (important for papers)
    writer.add_scalar("Loss/epoch", epoch_loss / n_batches, epoch)
    #writer.add_scalar("Accuracy/epoch", epoch_acc / n_batches * 100, epoch)
writer.close()