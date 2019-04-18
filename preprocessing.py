import os

import nibabel as nib
import numpy as np
import math


def add_rice_noise(img, snr=10, mu=0.0, sigma=1):
    level = snr * np.max(img) / 100
    size = img.shape
    x = level * np.random.normal(mu, sigma, size=size) + img
    y = level * np.random.normal(mu, sigma, size=size)
    return np.sqrt(x**2 + y**2).astype(np.int16)

# 给图像加噪声


def generate_noised_mri(noisy_level):
    files = os.listdir("./data/dataset/Free/")
    for file in files:
        nii_img = nib.load("./data/dataset/Free/" + file)
        free_image = nii_img.get_data()
        noised_image = add_rice_noise(free_image, snr=noisy_level)
        noised_image = nib.Nifti1Image(noised_image, nii_img.affine, nii_img.header)
        if not os.path.exists("./data/dataset/noise_%d/" % noisy_level):
            os.makedirs("./data/dataset/noise_%d/" % noisy_level)
        nib.save(noised_image, "./data/dataset/noise_%d/%s" % (noisy_level, file))


'''
切 patch
 32 * 32
 步长 5
一张256*256图产生56*56张patch
'''
def generate_patch(level):
    stride = 8
    size = 32
    depth = 6
    num = 0
    files = os.listdir("./data/dataset/Free/")
    for file in files:
        free_img = nib.load("./data//dataset/Free/" + file).get_data()
        noised_img = nib.load("./data/dataset/noise_%d/%s" % (level, file)).get_data()
        free_img_set = None
        noised_img_set = None
        height, width, _ = free_img.shape
        for y in range(0, height-size, stride):
            for x in range(0, width-size, stride):
                free_img_temp = free_img[y:y+size, x:x+size].copy().transpose(2, 0, 1)
                noised_img_temp = noised_img[y:y + size,
                                             x:x + size].copy().transpose(2, 0, 1)
                # print(free_img_temp.shape)
                free_img_temp = np.reshape(
                    free_img_temp, (1, 1, depth, size, size))
                noised_img_temp = np.reshape(
                    noised_img_temp, (1, 1, depth, size, size))
                
                if free_img_set is None:
                    free_img_set = free_img_temp
                    noised_img_set = noised_img_temp
                else:
                    free_img_set = np.append(free_img_set, free_img_temp, axis=0)
                    noised_img_set = np.append(noised_img_set, noised_img_temp, axis=0)
        num += 1
        print("-------" + str(num) + "-----------")
        print(noised_img_set.shape)
        print(free_img_set.shape)
        if not os.path.exists("./data/patchs32_32_%d/free/" % level):
            os.makedirs("./data/patchs32_32_%d/free/" % level)
            os.makedirs("./data/patchs32_32_%d/noised/" % level)
        np.save("./data/patchs32_32_%d/free/%d.npy" % (level, num), free_img_set)
        np.save("./data/patchs32_32_%d/noised/%d.npy" % (level, num), noised_img_set)

def patch_test_img(img, size=32):
    patchs = []
    height, width, depth = img.shape
    
    row = 0
    
    for i in range(0, height-size+1, size//2):
        row += 1
        col = 0
        for j in range(0, width-size+1, size//2):
            col += 1
            patchs.append(img[i:i+size, j:j+size,:])
    temp = np.vstack(patchs)
    temp = np.reshape(temp, (-1, size, size, depth))
    return temp, row, col

'''
N*H*W*D
'''


def merge_test_img(patchs, row, col,size=32):
    
    patchs_num = patchs.shape[0]
    num = int(math.sqrt(patchs_num))
    rows = []
    x = size // 8
    y = size//4
    row_index = 0
    for i in range(0, patchs_num, col):
        temp = patchs[i,:,:-x,:]
        for j in range(1, col - 1):
            temp[:, -y:, :] = (temp[ :, -y:, :] + patchs[i+j, :,x:x+y,:]) / 2
            temp = np.append(temp, patchs[i + j, :, x+y:-x, :], axis=1)
        temp[:, -y:, :] = (temp[:, -y:, :] + patchs[i+j+1, :,x:x+y,:]) / 2
        temp = np.append(temp, patchs[i + j+1, :, x+y:, :], axis=1)
        
        a = row_index * 16
        b = row_index * 16 + 32
        row_index += 1
        rows.append(temp)
    img = rows[0][:-x,:,:]
    length = len(rows)
    for i in range(1, length-1):
        height = img.shape[0]
        img[-y:, :, :] = (img[-y:, :, :] + rows[i][x:x+y, :, :])/2
        img = np.append(img, rows[i][x+y:-x, :, :], axis=0)
    img[-y:, :, :] = (img[ -y:, :, :] + rows[-1][x:x+y, :, :]) / 2
    img = np.append(img, rows[-1][ x+y:, :, :], axis=0)
        
    return img

    
if __name__ == "__main__":   
    # 第一步生成噪声图像
    generate_noised_mri(15)
    generate_patch(15)
    # sample = np.random.randn(256, 256, 10)
    # patchs = patch_test_img(sample)
    # print((patchs[14] - sample[:32, -32:, :]).sum())
    # print("----")
    # # print((patchs[0,:, -16:, :] - patchs[1, :, :16, :]).sum())
    # img = merge_test_img(patchs)

    # print((img - sample).sum())



