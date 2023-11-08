import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# Load dataset
from dataset import PTDataset
from unet_3D import UNet3D

for data_dir, filename_prefix in [("Task02_Heart", "la"), ("Task06_Lung", "lung")]:
    print(data_dir)
    dataset_full = PTDataset(data_dir, filename_prefix=filename_prefix, train_or_test="train")
    image, label = dataset_full[0]
    print(image.shape, label.shape)
    # x, y = dataset_full
    # print(x, y)
    print(len(dataset_full))