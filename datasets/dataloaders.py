import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Image-Mask Dataset Class
class ImageMaskDataset(Dataset):
    def __init__(
        self,
        root_dir,
        dataset_name,
        transform,
        mask_transform,
        split="train",
        mask_suffix="",
        mask_ext=".png",
    ):
        # Handle Oxford-IIIT Pet dataset case
        if dataset_name == "PET" and split in ["train", "val"]:
            split_dir = "trainval"  # Use 'trainval' for both train and val splits for PET dataset
        else:
            split_dir = split  # Use 'train', 'test', or 'val' for other datasets

        # Set the directories based on the split
        self.image_dir = os.path.join(root_dir, split_dir, "images")
        self.mask_dir = os.path.join(root_dir, split_dir, "masks") if "test" not in split else None
        self.dataset_name = dataset_name
        self.split = split
        self.mask_suffix = mask_suffix
        self.mask_ext = mask_ext
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = sorted(os.listdir(self.image_dir))

        # Load the label mapping from the JSON file for train/val splits
        if self.mask_dir:
            label_mapping_path = os.path.join(self.mask_dir, f"{split.capitalize()}_label_mapping.json")
            with open(label_mapping_path, "r") as json_file:
                self.mask_labels = json.load(json_file)
            # Normalize RGB values (if needed)
            self.normalized_mask_labels = {
                key: [float(val) / 255.0 for val in value] for key, value in self.mask_labels.items()
            }
            self.num_classes = len(self.normalized_mask_labels)

    def __len__(self):
        return len(self.image_files)

    def get_label_mapping(self):
        return [(key, value) for key, value in self.normalized_mask_labels.items()]

    def __getitem__(self, idx):
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
        # Open the mask and determine if it is RGB or grayscale
        mask = Image.open(mask_path)
        mask = self.mask_transform(mask)
        if mask.shape[0] == 3:
            # Convert RGB mask to integer-labeled mask
            mask_np = np.array(mask)
            mask_labeled = self.rgb_to_label(mask_np)
        else:
            # Grayscale masks are already labeled correctly
            mask_np = np.array(mask)
            mask_labeled = self.grayscale_to_label(mask_np)
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Convert mask to PyTorch tensor and ensure it is of type long
        mask_labeled = torch.tensor(mask_labeled, dtype=torch.long)

        # One-hot encode the mask and ensure the shape is correct
        one_hot_mask = F.one_hot(mask_labeled, num_classes=self.num_classes)
        one_hot_mask = one_hot_mask.permute(2, 0, 1)  # (num_classes, H, W)

        return image, one_hot_mask  # No need for unnecessary squeezing or unsqueezing

    def rgb_to_label(self, mask_np):
        """Convert an RGB mask to an integer-labeled mask."""
        label_mask = np.zeros((mask_np.shape[1], mask_np.shape[2]), dtype=np.int64)
        mask_np = np.transpose(mask_np, (1, 2, 0))
        for idx, (class_name, rgb_value) in enumerate(self.normalized_mask_labels.items()):
            # Check where the mask matches the current class's RGB value
            matches = np.all(mask_np == (np.array(rgb_value)).astype(np.uint8), axis=-1)
            label_mask[matches] = idx

        return label_mask

    def grayscale_to_label(self, mask_np):
        """Convert a grayscale mask (1, H, W) to a 2D integer-labeled mask (H, W) based on label mapping."""
        # Remove the channel dimension if it exists (from (1, H, W) -> (H, W))
        if len(mask_np.shape) == 3 and mask_np.shape[0] == 1:
            mask_np = np.squeeze(mask_np, axis=0)

        # Initialize a label mask with the same shape as the grayscale mask
        label_mask = np.zeros_like(mask_np, dtype=np.int64)

        # Map grayscale values to class indices using the label mapping
        for idx, (class_name, gray_value) in enumerate(self.normalized_mask_labels.items()):
            # Assign the class index where the grayscale value matches
            label_mask[mask_np == int(gray_value[0])] = idx

        return torch.tensor(label_mask, dtype=torch.long)


# Dataloader Creation Function
def create_dataloader(
    root_dir,
    dataset_name,
    split="train",
    batch_size=8,
    shuffle=True,
    img_size=(360, 640),
    mask_suffix="",
    mask_ext=".png",
    num_workers=2,
):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    dataset = ImageMaskDataset(
        root_dir,
        dataset_name,
        split=split,
        transform=transform,
        mask_transform=mask_transform,
        mask_suffix=mask_suffix,
        mask_ext=mask_ext,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


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

    pets_loader_test = create_dataloader(pets_root_dir, "PET", split="test", img_size=fixed_size)

    # MSD Brain Tumor Dataset Loaders
    brain_root_dir = "msd_brain"
    brain_loader_train = create_dataloader(brain_root_dir, "BRAIN", split="train", img_size=fixed_size)

    brain_loader_test = create_dataloader(brain_root_dir, "BRAIN", split="test", img_size=fixed_size)

    # MSD Heart Dataset Loaders
    heart_root_dir = "msd_heart"
    heart_loader_train = create_dataloader(heart_root_dir, "HEART", split="train", img_size=fixed_size)

    heart_loader_test = create_dataloader(heart_root_dir, "HEART", split="test", img_size=fixed_size)

    # Verifying the Dataloader Outputs
    def verify_dataloader(dataloader, name, num_samples=4):
        for batch in dataloader:
            # Check if it's a list and contains images + masks (train/val case)
            if isinstance(batch, list) and len(batch) == 2:
                images, one_hot_masks = batch
                print(f"{name} Dataloader: Image batch shape: {images.shape}, Mask batch shape: {one_hot_masks.shape}")

                # Assertions to verify one-hot encoding is valid
                assert (
                    one_hot_masks.dim() == 4
                ), "Expected one-hot masks to have 4 dimensions (batch, num_classes, H, W)."
                assert one_hot_masks.sum(dim=1).max() == 1, "Invalid one-hot encoding: More than one class per pixel."
                assert (
                    one_hot_masks.sum(dim=1).min() == 1
                ), "Invalid one-hot encoding: No class assigned to some pixels."

            else:  # Test case: only images
                images = batch
                print(f"{name} Dataloader: Image batch shape: {len(images)} images")

            # Display a few samples
            fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))

            for i in range(min(num_samples, len(images))):
                # Display the image
                img = images[i].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
                axs[i, 0].imshow(img)
                axs[i, 0].set_title("Image")
                axs[i, 0].axis("off")

                # Display the mask for train/val
                if isinstance(batch, list) and len(batch) == 2:
                    # Convert one-hot mask back to class index mask using argmax
                    mask = one_hot_masks[i].argmax(dim=0).numpy()  # (H, W)
                    axs[i, 1].imshow(mask, cmap="tab20")  # Use a categorical colormap
                    axs[i, 1].set_title("Class Index Mask")
                    axs[i, 1].axis("off")

            plt.tight_layout()
            plt.show()
            break  # Only verify the first batch

    # Verify each dataloader
    verify_dataloader(coco_loader_train, "COCO Train")
    verify_dataloader(coco_loader_val, "COCO Val")
    verify_dataloader(coco_loader_test, "COCO Test")
    verify_dataloader(voc_loader_train, "VOC Train")
    # verify_dataloader(voc_loader_val, "VOC Val")
    verify_dataloader(pets_loader_train, "Oxford-IIIT Pets Train")
    # verify_dataloader(pets_loader_test, "Oxford-IIIT Pets Test")
    verify_dataloader(brain_loader_train, "MSD Brain Tumor Train")
    verify_dataloader(brain_loader_test, "MSD Brain Tumor Test")
    verify_dataloader(heart_loader_train, "MSD Heart Train")
    verify_dataloader(heart_loader_test, "MSD Heart Test")
