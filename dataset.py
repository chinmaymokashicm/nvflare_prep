import os
import torch
import numpy as np
import nibabel as nib
from torchvision import transforms
from torch.utils.data import Dataset

class PTDataset(Dataset):
    def __init__(self, data_dir, filename_prefix="lung", train_or_test="train", transform=None, crop_size=(256, 256, 64)):
        self.data_dir = data_dir
        self.transform = transform
        self.train_or_test = train_or_test
        self.dict_subfolder_suffix = {"train": "Tr", "test": "Ts"}
        self.filename_prefix = filename_prefix
        self.filepaths_images = sorted([os.path.join(data_dir, f"images{self.dict_subfolder_suffix[train_or_test]}", f) for f in os.listdir(os.path.join(self.data_dir, f"images{self.dict_subfolder_suffix[train_or_test]}")) if f.startswith(self.filename_prefix)])
        if train_or_test == "train":
            self.filepath_labels = sorted([os.path.join(data_dir, "labelsTr", f) for f in os.listdir(os.path.join(self.data_dir, "labelsTr")) if f.startswith(self.filename_prefix)])
        self.crop_size = crop_size

    def load_nifti(self, filepath_data):
        img = nib.load(filepath_data)
        data = img.get_fdata()
        return data

    def crop_volume(self, volume):
        target_height, target_width, target_depth = self.crop_size
        print("Target:", target_height, target_width, target_depth)
        height, width, depth = volume.shape
        
        if target_height > height:
            target_height = height

        if target_width > width:
            target_width = width

        if target_depth > depth:
            target_depth = depth

        # Randomly select top-left coordinates for cropping
        top = torch.randint(0, height - target_height + 1, (1,))
        left = torch.randint(0, width - target_width + 1, (1,))
        depth_start = torch.randint(0, depth - target_depth + 1, (1,))

        # Crop the volume
        volume = volume[top:top + target_height, left:left + target_width, depth_start:depth_start + target_depth]
        return volume

    def pad_or_crop_volume(self, volume, crop_size=None):
        if crop_size is None:
            crop_size = self.crop_size
        target_height, target_width, target_depth = crop_size
        # print("Target dimensions:", target_height, target_width, target_depth)
        height, width, depth = volume.shape

        if height < target_height:
            # Pad image if it's smaller than the target size
            pad_height = target_height - height
            volume = np.pad(volume, ((0, pad_height), (0, 0), (0, 0)), mode="constant")
            height = target_height

        if width < target_width:
            # Pad image if it's smaller than the target size
            pad_width = target_width - width
            volume = np.pad(volume, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
            width = target_width

        if depth < target_depth:
            # Pad image if it's smaller than the target size
            pad_depth = target_depth - depth
            volume = np.pad(volume, ((0, 0), (0, 0), (0, pad_depth)), mode="constant")
            depth = target_depth

        # Randomly select top-left coordinates for cropping
        top = torch.randint(0, height - target_height + 1, (1,))
        left = torch.randint(0, width - target_width + 1, (1,))
        depth_start = torch.randint(0, depth - target_depth + 1, (1,))

        # Crop the volume
        volume = volume[top:top + target_height, left:left + target_width, depth_start:depth_start + target_depth]
        return volume

    def normalize(self, volume):
        # Normalize the volume values
        min_vals = volume.min(axis=(0, 1, 2))
        max_vals = volume.max(axis=(0, 1, 2))
        volume = (volume - min_vals) / (max_vals - min_vals)
        return volume

    def __len__(self):
        return len(self.filepaths_images)

    def __getitem__(self, idx):
        image = self.load_nifti(self.filepaths_images[idx])

        # Crop the 3D volume
        # print("Image current shape:", image.shape)
        # image = self.crop_volume(image)
        image = self.pad_or_crop_volume(image)
        # print("Image modified shape:", image.shape)

        # Normalize and convert to tensors
        image = self.normalize(image)
        image = torch.Tensor(image).unsqueeze(0)  # Add a batch dimension
        image = image.float()

        dict_output = {"image": image}

        if self.train_or_test == "train":
            label = self.load_nifti(self.filepath_labels[idx])
            label = self.pad_or_crop_volume(label, crop_size=(258, 258, 66))
            # label = torch.Tensor(label).unsqueeze(0)
            label = torch.Tensor(label)
            label = label.float()
            dict_output["label"] = label

        if self.transform:
            dict_output = {key: self.transform(value) for key, value in dict_output.items()}

        if self.train_or_test == "train":
            return dict_output["image"], dict_output["label"]
        else:
            return dict_output["image"]
