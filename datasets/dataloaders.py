import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Image-Mask Dataset Class
class ImageMaskDataset(Dataset):
    """
    Dataset class for loading images and corresponding masks for segmentation tasks.

    Attributes:
        image_dir (str): Directory path containing the images.
        mask_dir (str): Directory path containing the masks (None for test set).
        dataset_name (str): Name of the dataset (e.g., COCO, VOC).
        split (str): Dataset split type ('train', 'val', 'test').
        mask_suffix (str): Suffix to add to mask file names if needed.
        mask_ext (str): Extension for mask files (e.g., .png, .npy).
        transform (torchvision.transforms.Compose): Transformations to apply to images.
        size (tuple): Size (height, width) to resize images and masks to.
        mask_labels (dict): Mapping from class labels to class names (for train/val sets).
    """

    def __init__(self, root_dir, dataset_name, transform, size, split="train", mask_suffix="", mask_ext=".png"):
        """
        Initialize the ImageMaskDataset with directory paths and transformation settings.

        Args:
            root_dir (str): Root directory path for the dataset.
            dataset_name (str): Name of the dataset (e.g., COCO, VOC).
            transform (torchvision.transforms.Compose): Transformations for images.
            size (tuple): Desired image size (height, width).
            split (str): Split of the dataset ('train', 'val', 'test').
            mask_suffix (str): Optional suffix for mask filenames.
            mask_ext (str): Extension of mask files (default .png).
        """
        # Handle Oxford-IIIT Pet dataset case
        split_dir = split  # Use 'train', 'test', or 'val' for other datasets

        # Set the directories based on the split
        self.image_dir = os.path.join(root_dir, split_dir, "images")
        self.mask_dir = os.path.join(root_dir, split_dir, "masks") if "test" not in split else None
        self.dataset_name = dataset_name
        self.split = split
        self.mask_suffix = mask_suffix
        self.mask_ext = mask_ext
        self.transform = transform
        self.size = size
        self.image_files = sorted(os.listdir(self.image_dir))

        # Load the label mapping from the JSON file for train/val splits
        if self.mask_dir:
            label_mapping_path = os.path.join(self.mask_dir, f"{split.capitalize()}_label_mapping.json")
            with open(label_mapping_path, "r") as json_file:
                self.mask_labels = json.load(json_file)

            self.num_classes = len(self.mask_labels)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_files)

    def get_label_mapping(self):
        """Return the label mapping as a list of tuples (class, label)."""
        return [(key, value) for key, value in self.mask_labels.items()]

    def random_transform(self, image, mask):
        """
        Apply random horizontal and vertical flips to the image and mask for data augmentation.

        Args:
            image (PIL Image): Input image.
            mask (torch.Tensor): Corresponding mask.

        Returns:
            Tuple[Image, Tensor]: Transformed image and mask.
        """
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        # image = TF.to_tensor(image)
        # mask = TF.to_tensor(mask)

        return image, mask

    def __getitem__(self, idx):
        """
        Load an image and its corresponding mask, apply transformations, and return them.

        Args:
            idx (int): Index of the image to load.

        Returns:
            Tuple[Tensor, Tensor] or Tensor: Transformed image and mask for train/val; only the image for test.
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Open the image and convert to RGB
        image = Image.open(img_path).convert("RGB")

        # Handle test split: return only the transformed image
        if self.split == "test":
            if self.transform:
                image = self.transform(image)
            return image

        # For train/val: Load the mask
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}{self.mask_suffix}{self.mask_ext}"
        mask_path = os.path.join(self.mask_dir, mask_name)
        # Load the mask from the .npy file
        mask_np = np.load(mask_path)

        # Apply mask transformations
        resized_mask = cv2.resize(mask_np, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

        # Convert the mask to a PyTorch tensor
        mask_tensor = torch.tensor(resized_mask, dtype=torch.long)

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        if self.split == "train":
            image, mask_tensor = self.random_transform(image, mask_tensor)

        return image, mask_tensor  # No need for unnecessary squeezing or unsqueezing


# Dataloader Creation Function
def create_dataloader(
    root_dir,
    dataset_name,
    split="train",
    batch_size=8,
    shuffle=True,
    img_size=(360, 640),
    mask_suffix="",
    mask_ext=".npy",
    num_workers=2,
):
    """
    Create a DataLoader for the specified dataset and split.

    Args:
        root_dir (str): Root directory path for the dataset.
        dataset_name (str): Name of the dataset.
        split (str): Split of the dataset ('train', 'val', 'test').
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        img_size (tuple): Desired image size (height, width).
        mask_suffix (str): Suffix for mask filenames.
        mask_ext (str): Extension of mask files.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: DataLoader for the specified dataset and split.
    """
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

    dataset = ImageMaskDataset(
        root_dir,
        dataset_name,
        split=split,
        transform=transform,
        mask_suffix=mask_suffix,
        mask_ext=mask_ext,
        size=img_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def verify_dataloader(dataloader, name, num_samples=4):
    """
    Verify the DataLoader output shapes for a given dataset split.

    Args:
        dataloader (DataLoader): DataLoader to verify.
        name (str): Name of the dataset split (e.g., "COCO Train").
        num_samples (int): Number of samples to check.
    """
    for batch in dataloader:
        # Check if it's a list and contains images + masks (train/val case)
        if isinstance(batch, list) and len(batch) == 2:  # train/val split
            images, masks = batch
            print(f"{name} Dataloader: Image batch shape: {images.shape}, Mask batch shape: {masks.shape}")

            # Display only the specified number of samples
            for i in range(min(num_samples, len(images))):
                print(f"  Sample {i + 1} - Image shape: {images[i].shape}, Mask shape: {masks[i].shape}")
        else:  # test split, only images
            images = batch
            print(f"{name} Dataloader: Image batch shape: {len(images)} images")

            # Display only the specified number of samples
            for i in range(min(num_samples, len(images))):
                print(f"  Sample {i + 1} - Image shape: {images[i].shape}")
        break  # Stop after the first batch


if __name__ == "__main__":
    # SAMPLE WORKFLOW

    os.chdir("data")

    # Set a fixed size for all images and masks
    fixed_size = (360, 640)  # (height, width)

    # COCO Dataset Loaders
    coco_root_dir = "coco"
    coco_loader_train = create_dataloader(coco_root_dir, "COCO", split="train", img_size=fixed_size)

    coco_loader_val = create_dataloader(coco_root_dir, "COCO", split="val", img_size=fixed_size)

    coco_loader_test = create_dataloader(coco_root_dir, "COCO", split="test", img_size=fixed_size)

    # Pascal VOC Dataset Loaders, no test set
    voc_root_dir = "voc"
    voc_loader_train = create_dataloader(voc_root_dir, "VOC", split="train", img_size=fixed_size)

    voc_loader_val = create_dataloader(voc_root_dir, "VOC", split="val", img_size=fixed_size)

    # Oxford-IIIT Pet Dataset Loaders (trainval folder for both train and val)
    pets_root_dir = "oxford_pet"
    pets_loader_train = create_dataloader(pets_root_dir, "PET", split="train", img_size=fixed_size)

    pets_loader_val = create_dataloader(pets_root_dir, "PET", split="val", img_size=fixed_size)

    pets_loader_test = create_dataloader(pets_root_dir, "PET", split="test", img_size=fixed_size)

    # Oxford-IIIT Pet Dataset Loaders (trainval folder for both train and val)
    pets2_root_dir = "oxford_pet2"
    pets2_loader_train = create_dataloader(pets2_root_dir, "PET2", split="train", img_size=fixed_size)

    pets2_loader_val = create_dataloader(pets2_root_dir, "PET2", split="val", img_size=fixed_size)

    pets2_loader_test = create_dataloader(pets2_root_dir, "PET", split="test", img_size=fixed_size)

    # MSD Brain Tumor Dataset Loaders
    brain_root_dir = "msd_brain"
    brain_loader_train = create_dataloader(brain_root_dir, "BRAIN", split="train", img_size=fixed_size)

    brain_loader_test = create_dataloader(brain_root_dir, "BRAIN", split="test", img_size=fixed_size)

    # MSD Heart Dataset Loaders
    heart_root_dir = "msd_heart"
    heart_loader_train = create_dataloader(heart_root_dir, "HEART", split="train", img_size=fixed_size)

    heart_loader_test = create_dataloader(heart_root_dir, "HEART", split="test", img_size=fixed_size)

    # MSD Lung Dataset Loaders
    lung_root_dir = "msd_lung"
    lung_loader_train = create_dataloader(lung_root_dir, "LUNG", split="train", img_size=fixed_size)

    lung_loader_test = create_dataloader(lung_root_dir, "LUNG", split="test", img_size=fixed_size)

    # MSD Spleen Dataset Loaders
    spleen_root_dir = "msd_spleen"
    spleen_loader_train = create_dataloader(spleen_root_dir, "SPLEEN", split="train", img_size=fixed_size)

    spleen_loader_test = create_dataloader(spleen_root_dir, "SPLEEN", split="test", img_size=fixed_size)

    # Verify each dataloader
    verify_dataloader(coco_loader_train, "COCO Train")
    verify_dataloader(coco_loader_val, "COCO Val")
    verify_dataloader(coco_loader_test, "COCO Test")
    verify_dataloader(voc_loader_train, "VOC Train")
    verify_dataloader(voc_loader_val, "VOC Val")
    verify_dataloader(pets_loader_train, "Oxford-IIIT Pets Train")
    verify_dataloader(pets_loader_val, "Oxford-IIIT Pets Val")
    verify_dataloader(pets2_loader_train, "Oxford-IIIT Pets 2 Train")
    verify_dataloader(pets2_loader_val, "Oxford-IIIT Pets 2 Val")
    verify_dataloader(brain_loader_train, "MSD Brain Tumor Train")
    verify_dataloader(brain_loader_test, "MSD Brain Tumor Test")
    verify_dataloader(heart_loader_train, "MSD Heart Train")
    verify_dataloader(heart_loader_test, "MSD Heart Test")
    verify_dataloader(lung_loader_train, "MSD Lung Train")
    verify_dataloader(lung_loader_test, "MSD Lung Test")
    verify_dataloader(spleen_loader_train, "MSD Spleen Train")
    verify_dataloader(spleen_loader_test, "MSD Spleen Test")
