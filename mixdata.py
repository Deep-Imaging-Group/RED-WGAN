from random import shuffle

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MRIDataset(Dataset):
    def __init__(self, level):
        levels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        levels = [1, 2, 3, 4, 5]
        print("Start reading trainning dataset...")
        free_mri_set = np.load("./data/mixdata/free/%d.npy" % levels[0])
        noised_mri_set = np.load("./data/mixdata/noised/%d.npy" % levels[0])
        for level in levels[1:]:
            free_temp = np.load("./data/mixdata/free/%d.npy" % level)
            noised_temp = np.load("./data/mixdata/noised/%d.npy" % level)
            # print(noised_mri_set.shape, noised_temp.shape)
            free_mri_set = np.append(free_mri_set, free_temp, axis=0)
            noised_mri_set = np.append(noised_mri_set, noised_temp, axis=0)
        self.free_mri_set = free_mri_set
        self.noised_mri_set = noised_mri_set
        self.total = self.free_mri_set.shape[0]
        self.current_patch = 1
        self.indexs = list(range(self.total))
        shuffle(self.indexs)
        print(self.free_mri_set.shape)
        print("End reading trainning dataset...")

    def __len__(self):
        # return 90000
        # return 100
        return self.total

    def __getitem__(self, index):
        index = self.indexs[index]
        free_img = torch.from_numpy(self.free_mri_set[index]).float()
        noised_img = torch.from_numpy(self.noised_mri_set[index]).float()
        return {"free_img": free_img, "noised_img": noised_img}


class MRIValidDataset(Dataset):
    def __init__(self, level):
        print("Start reading testing dataset...")
        levels = [1, 2, 3,  4, 5, 7, 9, 11, 13, 15, 17, 19]
        free_mri_set = np.load("./data/mixdata/valid/free/%d.npy" % levels[0])
        noised_mri_set = np.load("./data/mixdata/valid/noised/%d.npy" % levels[0])
        for level in levels[1:]:
            free_temp = np.load("./data/mixdata/valid/free/%d.npy" % level)
            noised_temp = np.load("./data/mixdata/valid/noised/%d.npy" % level)

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
        # return 100
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


# data = MRIDataset()
# print(len(data))
# MRIValidDataset()
