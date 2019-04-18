import os
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import nibabel as nib
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

from preprocessing import patch_test_img, merge_test_img
from data import *

'''
Version: 0.0.1
Date: 2018-04-01
Structure:
    Generator: Residual connection
        input  -> 32 -> 64 -> 128 -> 256 
        output <- 32 <- 64 <- 128 <——

        encoder layer: Conv3d -> BatchNorm3d -> LeakyReLu
        decoder layer: Conv3D - > Add encoder -> BatchNorm3d -> LeakyReLU(expect last layer is ReLu)
    Disciminator:
        input -> 32 -> 64 -> 128 -> 1
        Except last layer:Conv3d -> BatchNorm3d -> LeakyReLu
        Last layer: Full Connection(no active function)
'''


class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )
        # torch.nn.init.
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU()
        )

        self.deConv1_1 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.deConv1 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )

        self.deConv2_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.deConv2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )

        self.deConv3_1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.deConv3 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
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


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            # nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )

        self.fc = nn.Linear(128 * 6 * 32 * 32, 1)
        # self.fc2 = nn.Linear(1024, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        output = self.fc(x)
        # output = self.fc2(x)

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i

        return num_features


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.feature = vgg19.features
    '''
    input: N*1*D(6)*H*W
    output: N*C*H*W
    '''

    def forward(self, input):
        # VGG19: means:103.939, 116.779, 123.68
        input /= 16
        depth = input.size()[2]
        result = []
        for i in range(depth):
            x = torch.cat(
                (input[:, :, i, :, :] - 103.939, input[:, :, i, :, :] - 116.779, input[:, :, i, :, :] - 123.68), 1)
            result.append(self.feature(x))

        output = torch.cat(result, dim=1)

        # output = self.feature(input)

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


class WGAN():
    def __init__(self, level):
        # parameters
        self.epochs = 10
        # self.batch_size = 120
        self.batch_size = 110
        self.lr =5e-6
        #  self.lr = 0.0000125000

        self.d_iter = 5
        self.lambda_gp = 10
        # self.lambda_vgg = 1e-3
        # self.lambda_d = 1e-5

        self.lambda_vgg = 1e-1
        self.lambda_d = 1e-3
        self.lambda_mse = 1

        self.level = level

        self.loss_dir = "./loss/"
        # self.v = "0_0_1_15" # ter
        self.v = "0_0_5_%d" % self.level  # vs
        self.save_dir = "./model/" + self.v + "/"
        self.gpu = False

        self.generator = GeneratorNet()
        self.discriminator = DiscriminatorNet()
        self.vgg19 = VGG19()

        self.G_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))

        # self.G_optimizer = optim.RMSprop(self.generator.parameters())
        # self.D_optimizer = optim.RMSprop(self.discriminator.parameters())

        self.G_loss = nn.MSELoss()

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.vgg19.cuda()
            self.gpu = True
        if not self.load_model():
            initialize_weights(self.generator)
            initialize_weights(self.discriminator)

    def train(self):
        self.dataset = MRIDataset(self.level)
        # self.dataset = MRIDataset10125()
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)

        self.validDataset = MRIValidDataset(self.level)
        self.validDataloader = DataLoader(
            self.validDataset, batch_size=self.batch_size, shuffle=True)
        # 训练次数
        for epoch in range(0, self.epochs):
            self.test(epoch)
            # 迭代数据集
            for batch_index, batch in enumerate(self.dataloader):
                # if (batch_index % 100 == 0):
                #     self.test(epoch)
                # print("epoch:", epoch, ";batch number:", batch_index, ";D_Loss:", end="")
                free_img = batch["free_img"]
                noised_img = batch["noised_img"]
                # print(type(noised_img))

                # 训练discriminator
                for iter_i in range(self.d_iter):
                    loss = self._train_discriminator(free_img, noised_img)
                    print("\tVGG_MSE - lr: %.10f, Level: %d, Epoch: %d, bath_index: %d, iter: %d, G-Loss: " %
                          (self.lr, self.level, epoch, batch_index, iter_i), loss)

                # 训练genarator
                loss = self._train_generator(free_img, noised_img)
                # print("G Loss:%.4f, %.4f" %
                #    (float(loss[0]), float(loss[1])))

                # 保存模型和损失值
                if batch_index % 100 == 0:
                    self.save_model()


            if ((epoch + 1) % 4 == 0 and self.lr > 1e-7):
                self.G_optimizer.defaults["lr"] *= 0.5
                self.G_optimizer.defaults["lr"] *= 0.5
                self.lr *= 0.5

    def _train_discriminator(self, free_img, noised_img, train=True):
        self.D_optimizer.zero_grad()

        z = Variable(noised_img)
        real_img = Variable(free_img / 4096)
        if self.gpu:
            z = z.cuda()
            real_img = real_img.cuda()

        fake_img = self.generator(z)
        real_validity = self.discriminator(real_img)
        fake_validity = self.discriminator(fake_img.data / 4096)
        gradient_penalty = self._calc_gradient_penalty(
            real_img.data, fake_img.data)

        d_loss = torch.mean(-real_validity) + torch.mean(fake_validity) + \
            self.lambda_gp * gradient_penalty
        if train:
            d_loss.backward()
            # torch.mean(-real_validity).backward()
            # (torch.mean(-real_validity) + torch.mean(fake_validity)).backward()
            # torch.mean(-real_validity).backward()
            # torch.mean(fake_validity).backward()
            self.D_optimizer.step()

        return d_loss.data.item(), torch.mean(-real_validity).cpu().item(), torch.mean(fake_validity).cpu().item(), self.lambda_gp * gradient_penalty.cpu().item()

    def _train_generator(self, free_img, noised_img, train=True):
        z = Variable(noised_img)
        real_img = Variable(free_img, requires_grad=False)


        if self.gpu:
            z = z.cuda()
            real_img = real_img.cuda()

        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()
        self.vgg19.zero_grad()

        criterion_mse = nn.MSELoss()
        criterion_vgg= nn.MSELoss()

        fake_img = self.generator(z)
        mse_loss = criterion_mse(fake_img, real_img)
        if train:
            (self.lambda_mse * mse_loss).backward(retain_graph=True)


        feature_fake_vgg = self.vgg19(fake_img)
        feature_real_vgg = Variable(self.vgg19(real_img).data, requires_grad=False).cuda()

        vgg_loss = criterion_vgg(feature_fake_vgg, feature_real_vgg)

        fake_validity = self.discriminator(fake_img / 4096)
        # g_loss = self.lambda_mse * mse_loss + self.lambda_vgg * vgg_loss + self.lambda_d * torch.mean(-fake_validity)
        g_loss =  self.lambda_vgg * vgg_loss + self.lambda_d * torch.mean(-fake_validity)

        if train:
            # (self.lambda_mse * mse_loss).backward()
            g_loss.backward()
            self.G_optimizer.step()
        return g_loss.data.item(), mse_loss.data.item(), torch.mean(-fake_validity).data.item(), vgg_loss.data.item()

    def _calc_gradient_penalty(self, free_img, gen_img):
        batch_size = free_img.size()[0]
        alpha = Variable(torch.rand(batch_size, 1))
        alpha = alpha.expand(batch_size, free_img.nelement(
        ) // batch_size).contiguous().view(free_img.size()).float()
        if self.gpu:
            alpha = alpha.cuda()

        interpolates = (alpha * free_img + (1 - alpha)
                        * gen_img).requires_grad_(True)
        disc_interpolates = self.discriminator(interpolates)
        fake = Variable(torch.Tensor(batch_size, 1).fill_(1.0),
                        requires_grad=False)
        if self.gpu:
            fake = fake.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]

        # gradients = gradients.view(gradients.size(0), -1)
        # print(gradients.size())
        # print(torch.norm(gradients, 2, dim=1).size())

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        # print("gradient_penalty: ", gradient_penalty.cpu().item())
        return gradient_penalty

    def test(self, epoch):
        timestr = time.strftime("%H:%M:%S", time.localtime())
        print(timestr)
        # 测试集
        total_mse_loss = 0
        total_g_loss = 0
        total_d_loss = 0
        total_vgg_loss = 0
        batch_num = 0
        for batch_index, batch in enumerate(self.validDataloader):
            free_img = batch["free_img"]
            noised_img = batch["noised_img"]

            loss = self._train_generator(free_img, noised_img, train=False)
            # print(loss, end=" ;")
            total_g_loss += loss[0]
            total_mse_loss += loss[1]
            total_d_loss += loss[2]
            total_vgg_loss += loss[3]
            batch_num += 1
        mse_loss = total_mse_loss / batch_num
        g_loss = total_g_loss / batch_num
        d_loss = total_d_loss / batch_num
        vgg_loss = total_vgg_loss / batch_num
        print("%s Epoch： %d lr：%.10f Test Loss：g-loss: %.4f vgg-loss: %.4f mse-loss： %.4f d_loss: %.4f" %
                (self.v, epoch, self.G_optimizer.defaults["lr"], g_loss, vgg_loss, mse_loss, d_loss))
        self.compute_quality()
        self.save_loss((vgg_loss, mse_loss, g_loss, d_loss))
        self.save_model()

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
        for i in range(101, 111):
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
        timestr = time.strftime("%H:%M:%S", time.localtime())
        with open("./loss/" + self.v + "psnr_4096.csv", "a+") as f:
            f.write("%s: %.10f,%f,%f,%f,%f,%f,%f\n" %
                    (timestr, self.lr, psnr1, psnr2, ssim1, ssim2, mse1, mse2))
        print("psnr: %f,%f,ssim: %f,%f,mse:%f,%f\n" %
              (_psnr1, _psnr2, _ssim1, _ssim2, _mse1, _mse2))

    '''
    N*H*W*D -> N*C*D*H*W
    return: N*H*W*D
    '''

    def denoising(self, patchs):
        n, h, w, d = patchs.shape
        denoised_patchs = []
        for i in range(0, n, self.batch_size):
            batch = patchs[i:i + self.batch_size]
            batch_size = batch.shape[0]
            x = np.reshape(batch, (batch_size, 1, w, h, d))
            x = x.transpose(0, 1, 4, 2, 3)
            x = Variable(torch.from_numpy(x).float()).cuda()
            y = self.generator(x)
            denoised_patchs.append(y.cpu().data.numpy())
        # print(len(denoised_patchs))
        denoised_patchs = np.vstack(denoised_patchs)
        # print(denoised_patchs.shape)
        denoised_patchs = np.reshape(denoised_patchs, (n, d, h, w))
        denoised_patchs = denoised_patchs.transpose(0, 2, 3, 1)
        # print(denoised_patchs.shape)
        return denoised_patchs

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.generator.state_dict(),
                   self.save_dir + "G_" + self.v + ".pkl")
        torch.save(self.discriminator.state_dict(),
                   self.save_dir + "D_" + self.v + ".pkl")

    def load_model(self):
        if os.path.exists(self.save_dir + "G_" + self.v + ".pkl") and \
                os.path.exists(self.save_dir + "D_" + self.v + ".pkl"):

            self.generator.load_state_dict(
                torch.load(
                    self.save_dir + "G_" + self.v + ".pkl",
                    map_location={'cuda:1': 'cuda:0'}))
            self.discriminator.load_state_dict(
                torch.load(
                    self.save_dir + "D_" + self.v + ".pkl",
                    map_location={'cuda:1': 'cuda:0'}))
            return True
        else:
            return False

    def save_loss(self, loss):
        value = ""
        for item in loss:
            value = value + str(item) + ","
        value += "\n"
        with open("./loss/" + self.v + ".csv", "a+") as f:
            f.write(value)


if __name__ == "__main__":
    # torch.cuda.set_device(1)
    for level in range(1, 4, 2):
        # level = 15
        wgan = WGAN(level)
        # training
        wgan.train()

    # valid
    for i in range(101, 111):

        nii_img = nib.load("./data/dataset/noise_%d/%d.nii" % (level, i))
        x = nii_img.get_data()
        patchs, row, col = patch_test_img(x)
        print(patchs.shape)
        denoised_img = merge_test_img(wgan.denoising(patchs), row, col)
        print(denoised_img.shape)
        denoised_img = denoised_img.astype(np.int16)

        denoised_image = nib.Nifti1Image(
            denoised_img, nii_img.affine, nii_img.header)
        nib.save(denoised_image, "./result/%d_wgan_vgg_mse_denoised_img%d.nii" % (level, i))
    # print((x - denoised_img).mean())
