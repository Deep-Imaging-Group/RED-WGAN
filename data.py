import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, level):
        print("Start reading trainning dataset...")
        print("./data/patchs32_32_%d" % level)
        free_mri_set = np.load("./data/patchs32_32_%d/free/1.npy" % level)
        noised_mri_set = np.load("./data/patchs32_32_%d/noised/1.npy" % level)
        for i in range(2, 100):
            path1 = "./data/patchs32_32_%d/free/%d.npy" % (level, i)
            path2 = "./data/patchs32_32_%d/noised/%d.npy" % (level, i)
            free_temp = np.load(path1)
            noised_temp = np.load(path2)

            free_mri_set = np.append(free_mri_set, free_temp, axis=0)
            noised_mri_set = np.append(noised_mri_set, noised_temp, axis=0)
        self.free_mri_set = free_mri_set
        self.noised_mri_set = noised_mri_set
        self.total = self.free_mri_set.shape[0]
        self.current_patch = 1
        print(self.free_mri_set.shape)
        print("End reading trainning dataset...")

    def __len__(self):
        # return 90000
        return self.total

    def __getitem__(self, index):
        free_img = torch.from_numpy(self.free_mri_set[index]).float()
        noised_img = torch.from_numpy(self.noised_mri_set[index]).float()
        return {"free_img": free_img, "noised_img": noised_img}


class MRIValidDataset(Dataset):
    def __init__(self, level):
        print("Start reading testing dataset...")
        print("./data/patchs32_32_%d" % level)
        free_mri_set = np.load("./data/patchs32_32_%d/free/101.npy" % level)
        noised_mri_set = np.load("./data/patchs32_32_%d/noised/101.npy" % level)
        for i in range(101, 111):
            path1 = "./data/patchs32_32_%d/free/%d.npy" % (level, i)
            path2 = "./data/patchs32_32_%d/noised/%d.npy" % (level, i)
            free_temp = np.load(path1)
            noised_temp = np.load(path2)

            free_mri_set = np.append(free_mri_set, free_temp, axis=0)
            noised_mri_set = np.append(noised_mri_set, noised_temp, axis=0)
        self.free_mri_set = free_mri_set
        self.noised_mri_set = noised_mri_set
        self.total = self.free_mri_set.shape[0]
        self.current_patch = 1
        print(self.free_mri_set.shape)
        print("End reading testing dataset...")

    def __len__(self):
        # return 90000
        return self.total

    def __getitem__(self, index):
        free_img = torch.from_numpy(self.free_mri_set[index]).float()
        noised_img = torch.from_numpy(self.noised_mri_set[index]).float()
        return {"free_img": free_img, "noised_img": noised_img}

def add_rice_noise(img, snr=1, mu=0.0, sigma=1):
    level = snr * np.max(img) / 100
    size = img.shape
    x = level * np.random.normal(mu, sigma, size=size) + img
    y = level * np.random.normal(mu, sigma, size=size)
    return np.sqrt(x**2 + y**2)
    # size = img.shape
    # x = snr * np.random.normal(mu, sigma, size=size) + img
    # y = snr * np.random.normal(mu, sigma, size=size)
    # return np.sqrt(x**2 + y**2)

# MRIDataset()
# MRIValidDataset()
