import os
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])


def getDataPath(data_path):
    with open(os.path.join(data_path, "tp_list.txt"), "r") as file:
        tp_lines = file.readlines()

    mask_path = []
    au_path = []
    tp_path = []

    for item in tp_lines:
        # mask_path
        parts = item.split('.')
        target_filename_mask = os.path.join(data_path, "Gt", ''.join(parts[0]) + '_gt.png')
        mask_path.append(target_filename_mask)

        # au_path 暂时没用到
        parts = item.split("_")
        target_filename_au = os.path.join(data_path, "Au", "Au_ani_" + "".join(parts[5]))
        au_path.append(target_filename_au)

        # tp_path
        parts = item.split("\n")
        target_filename_tp = os.path.join(data_path, "Tp", parts[0])
        tp_path.append(target_filename_tp)

    return tp_path, mask_path


def show_masks(masks):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.imshow(masks[i], cmap='gray')
        ax.set_title(f'Mask {i + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class Dataset():
    cmap = voc_cmap()
    def __init__(self, tp_paths, mask_paths, transform=None):
        self.tp_paths = tp_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.tp_paths)

    def __getitem__(self, idx):
        tp_img = cv2.imread(self.tp_paths[idx], 1)
        tp_img = cv2.resize(tp_img, (256, 256))
        tp_img = tp_img / 255.0
        tp_img = np.moveaxis(tp_img, 2, 0)
        tp_img = np.float32(tp_img)
        tp_img = torch.from_numpy(tp_img)

        mask_img = cv2.imread(self.mask_paths[idx], 0)
        mask_img = cv2.resize(mask_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_img = mask_img / 255.0

        # mask_img = torch.from_numpy(mask_img)
        return tp_img, mask_img
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


if __name__ == '__main__':
    data_path = "/root/autodl-tmp/dataset/casia"
    tp_path, mask_path = getDataPath(data_path)

    dataset = Dataset(tp_path, mask_path, transform)
    tp_img, mask_img = dataset[10]
    print(tp_img)
    print(mask_img[0], mask_img[0].shape)
    print(mask_img[1], mask_img[1].shape)
    print(mask_img[2], mask_img[2].shape)
    print(mask_img[3], mask_img[3].shape)
