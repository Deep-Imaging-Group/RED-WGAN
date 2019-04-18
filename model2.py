import os
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import nibabel as nib
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

from preprocessing import patch_test_img, merge_test_img
from data import *

'''
Version: 0.0.2
Date: 2018-04-09
Structure: CNN
        input  -> 32 -> 64 -> 128 -> 256 
        output <- 32 <- 64 <- 128 <——

        encoder layer: Conv3d -> BatchNorm3d -> LeakyReLu
        decoder layer: Conv3D - > Add encoder -> BatchNorm3d -> LeakyReLU(expect last layer is ReLu)

'''


class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True)
        )
        # torch.nn.init.
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.deConv1_1 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.deConv1 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True)
        )

        self.deConv2_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.deConv2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.deConv3_1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.deConv3 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.deConv4_1 = nn.Conv3d(32, 1, kernel_size=3, padding=1)

        self.deConv4 = nn.ReLU()

    def forward(self, input):
        conv1 = self.conv1(input)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x = self.deConv1_1(conv4)
        x = x + conv3

        deConv1 = self.deConv1(x)

        x = self.deConv2_1(deConv1)
        x += conv2
        deConv2 = self.deConv2(x)

        x = self.deConv3_1(deConv2)
        x += conv1
        deConv3 = self.deConv3(x)

        x = self.deConv4_1(deConv3)
        x += input
        output = self.deConv4(x)

        return output


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Net():
    def __init__(self, level):
        self.lr = 5e-4
        self.epochs = 50
        # self.level = 3
        self.level = level
        self.v = "0_0_2_%d" % self.level
        # self.v = "0_0_2_13_64_64"
        # self.v = "0_0_2_13_sgd"
        self.cnn3d = CNN3D().cuda()
        initialize_weights(self.cnn3d)
        self.batch_size = 100
        self.save_dir = "./model/" + self.v + "/"

        # self.dataset = MRIDataset()

        self.load_model()

    def train(self):
        self.dataset = MRIDataset(self.level)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size)

        self.validDataset = MRIValidDataset(self.level)
        self.validDataloader = DataLoader(
            self.validDataset, batch_size=self.batch_size)
        print("Start trainning...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.cnn3d.parameters(), lr=self.lr, betas=(0.9, 0.99))
        # optimizer = optim.SGD(self.cnn3d.parameters(), lr=self.lr)

        for epoch in range(0,self.epochs):
            timestr = time.strftime("%H:%M:%S", time.localtime())
            print(timestr)
            print("CNN - Level: %d, Epoch %d lr： %.9f Test Loss:" %
                  (self.level,epoch, optimizer.defaults["lr"]), end="")
            batch_num = 0
            total_loss = 0
            for batch_index, batch in enumerate(self.validDataloader):
                target = Variable(batch["free_img"]).cuda()
                x = Variable(batch["noised_img"]).cuda()
                y = self.cnn3d(x)
                # print(loss, end=" ;")
                loss = criterion(y, target)
                total_loss += loss.item()
                # total_loss += loss.data[0]
                batch_num += 1
            loss = total_loss / batch_num
            print("%.4f" % (loss))

            self.compute_quality()
            for batch_index, batch in enumerate(self.dataloader):
                free_img = batch["free_img"]
                noised_img = batch["noised_img"]

                x = Variable(noised_img).cuda()
                target = Variable(free_img).cuda()

                y = self.cnn3d(x)
                loss = criterion(y, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("Epoch: ", epoch, "Batch: ",
                #       batch_index, "Loss: ", loss.data[0])



            # 保存模型和损失值
            self.save_model()
            self.save_loss(total_loss / batch_num)
            if (epoch + 1) % 10 == 0 and self.lr > 1e-7:
                optimizer.defaults["lr"] *= 0.1
                self.lr *= 0.1

    def denoising(self, patchs):
        n, h, w, d = patchs.shape
        denoised_patchs = []
        for i in range(0, n, self.batch_size):
            batch = patchs[i:i + self.batch_size]
            batch_size = batch.shape[0]
            x = np.reshape(batch, (batch_size, 1, w, h, d))
            x = x.transpose(0, 1, 4, 2, 3)
            x = Variable(torch.from_numpy(x).float()).cuda()
            y = self.cnn3d(x)
            denoised_patchs.append(y.cpu().data.numpy())
            # print(len(denoised_patchs))
        denoised_patchs = np.vstack(denoised_patchs)
        # print(denoised_patchs.shape)
        denoised_patchs = np.reshape(denoised_patchs, (n, d, h, w))

        denoised_patchs = denoised_patchs.transpose(0, 2, 3, 1)
        # print(denoised_patchs.shape)
        return denoised_patchs

    def compute_quality(self):
        psnr1 = 0
        psnr2 = 0
        mse1 = 0
        mse2 = 0
        ssim1 = 0
        ssim2 = 0
        _psnr1 = 0
        _psnr2 = 0
        _mse1 = 0
        _mse2 = 0
        _ssim1 = 0
        _ssim2 = 0
        for i in range(111, 121):
            free_nii = nib.load("./data/dataset/Free/%d.nii" % i)
            noised_nii = nib.load(
                "./data/dataset/noise_%d/%d.nii" % (self.level, i))

            free_img = free_nii.get_data()[:, :144, :].astype(np.int16)
            noised_img = noised_nii.get_data()[:, :144, :].astype(np.int16)
            patchs, row, col = patch_test_img(noised_img)
            denoised_img = merge_test_img(
                self.denoising(patchs), row, col).astype(np.int16)
            psnr1 += compare_psnr(free_img, noised_img, 4096)
            psnr2 += compare_psnr(free_img, denoised_img, 4096)

            mse1 += compare_nrmse(free_img, noised_img)
            mse2 += compare_nrmse(free_img, denoised_img)

            ssim1 += compare_ssim(free_img, noised_img,
                                  data_range=4096, multichannel=True)
            ssim2 += compare_ssim(free_img, denoised_img,
                                  data_range=4096, multichannel=True)

            max = np.max(free_img)
            _psnr1 += compare_psnr(free_img, noised_img, max)
            _psnr2 += compare_psnr(free_img, denoised_img, max)

            _mse1 += compare_nrmse(free_img, noised_img)
            _mse2 += compare_nrmse(free_img, denoised_img)

            _ssim1 += compare_ssim(free_img, noised_img,
                                   data_range=max, multichannel=True)
            _ssim2 += compare_ssim(free_img, denoised_img,
                                   data_range=max, multichannel=True)
        psnr1 *= 0.1
        psnr2 *= 0.1
        mse1 *= 0.1
        mse2 *= 0.1
        ssim1 *= 0.1
        ssim2 *= 0.1

        _psnr1 *= 0.1
        _psnr2 *= 0.1
        _mse1 *= 0.1
        _mse2 *= 0.1
        _ssim1 *= 0.1
        _ssim2 *= 0.1
        with open("./loss/" + self.v + "psnr.csv", "a+") as f:
            f.write("%f,%f,%f,%f,%f,%f\n" %
                    (_psnr1, _psnr2, _ssim1, _ssim2, _mse1, _mse2))
        with open("./loss/" + self.v + "psnr_4096.csv", "a+") as f:
            f.write("%f,%f,%f,%f,%f,%f,%f\n" %
                    (self.lr, psnr1, psnr2, ssim1, ssim2, mse1, mse2))
        print("psnr: %f,%f,ssim: %f,%f,mse:%f,%f\n" %
              (_psnr1, _psnr2, _ssim1, _ssim2, _mse1, _mse2))
    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.cnn3d.state_dict(),
                   self.save_dir + self.v + ".pkl")
        torch.save(self.cnn3d.state_dict(),
                   self.save_dir + self.v + ".pkl")

    def load_model(self):
        if os.path.exists(self.save_dir +  self.v + ".pkl"):
            self.cnn3d.load_state_dict(torch.load(
                self.save_dir  + self.v + ".pkl"))
        else:
            return False

    def save_loss(self, loss):
        value = str(loss)  + "\n"
        with open("./loss/" + self.v + ".csv", "a+") as f:
            f.write(value)


if __name__ == "__main__":
    for level in range(9, 16, 2):
        # level = 3
        net  = Net(level)
        net.train()

    # for i in range(101, 111):
    #     free_nii = nib.load("./data/dataset/Free/%d.nii" % i)
    #     nii_img = nib.load("./data/dataset/noise_%d/%d.nii" % (level, i))
    #     # ni i_img = nib.load("./data/patchs128_128_2d/img_noisy/130.nii")
    #     x = nii_img.get_data()
    #     free_img = free_nii.get_data()
    #     patchs, row, col = patch_test_img(x)
    #     meigred_img = merge_test_img(patchs, row, col)

    #     print(patchs.shape, row, col)
    #     print(meigred_img.shape)
    #     denoised_img = merge_test_img(net.denoising(patchs), row, col)
    #     height, width, depth = denoised_img.shape
    #     print(denoised_img.shape)
    #     # denoised_img = np.round(denoised_img)
    #     print(denoised_img.shape)
    #     # import matplotlib.pyplot as plt

    #     # plt.imshow(denoised_img[:, :, 1], cmap=plt.cm.bone)
    #     # plt.show()
    #     original_diff = (x - free_img)
    #     diff = (denoised_img - free_img[:, :width, :])

    #     diff_img = nib.Nifti1Image(free_img, nii_img.affine, nii_img.header)
    #     denoised_image = nib.Nifti1Image(
    #         denoised_img.astype(np.int16), nii_img.affine, nii_img.header)
    #     denoised_diff = nib.Nifti1Image(
    #         x, nii_img.affine, nii_img.header)
    #     nib.save(denoised_image,
    #              "./result/%d_model2_denoised_img_%d.nii" % (level, i))
    #     # nib.save(diff_img, "./result/model2_denoised_diff_img_%d.nii" % i)
    #     # nib.save(denoised_diff, "./result/model2_original_diff_img_%d.nii" % i)
    index = 1
    nii = nib.load("./realdata/%d_slice.nii" % index)
    data = nii.get_data()
    data = data[np.newaxis, np.newaxis, :, :, :]
    data = np.transpose(data, (0, 1, 4, 2, 3))
    print(data.shape)
    x = Variable(torch.from_numpy(data)).cuda().float()
    print(type(x))
    y = net.cnn3d(x)
    y = y.detach().cpu().numpy()[0, 0, :, :, :]
    y = np.transpose(y, (1, 2, 0))
    print(y.shape)
    denoised_img = nib.Nifti1Image(y, nii.affine, nii.header)
    nib.save(denoised_img, "./realdata/%d_slice_model2_denoised.nii" % index)
