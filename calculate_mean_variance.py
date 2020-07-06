import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets


#################### set batch size #####################

TRAIN_PATH = "./data/datasets/DIV2K/DIV2K_train_HR/"
batch_size = 50

train_tfms = transforms.Compose([
            transforms.ToTensor()
        ])
train_ds = datasets.ImageFolder(root=TRAIN_PATH, transform=train_tfms)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:

        b, c, h, w = images.shape
        # print(images.shape)
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


mean, std = online_mean_and_sd(train_dl)
print(mean, std/800)


###################################################################
# train_root = './data/datasets/DIV2K/DIV2K_train_HR/'
# trainset = torchvision.datasets.ImageFolder(
#     root=train_root,
#     transform=transforms.ToTensor()
# )
#
# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.dataset = data
#
#     def __getitem__(self, index):
#         x = self.dataset[index]
#         return x
#
#     def __len__(self):
#         return len(self.dataset)
#
#
# dataset = MyDataset(trainset)
# loader = DataLoader(
#     dataset,
#     batch_size=20,
#     num_workers=1,
#     shuffle=False
# )
#
# mean = 0.
# std = 0.
# nb_samples = 0.
# for data in loader:
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples
#
# mean /= nb_samples
# std /= nb_samples

