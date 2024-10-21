import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

os.chdir("data")


# Image-Mask Dataset Class
class ImageMaskDataset(Dataset):
    def __init__(
        self,
        root_dir,
        dataset_name,
        split="train",
        transform=None,
        mask_transform=None,
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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        # If it's a test split, return only the image
        if self.split == "test":
            if self.transform:
                image = self.transform(image)
            return image

        # For train and val, return image, mask, and labels
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}{self.mask_suffix}{self.mask_ext}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Handle masks depending on the dataset
        if "HEART" in self.dataset_name or "BRAIN" in self.dataset_name:
            mask = Image.open(mask_path).convert("L")  # Integer mask for these datasets
        else:
            mask = Image.open(mask_path).convert("RGB")  # For RGB-based datasets like COCO or VOC

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        labels = [(key, value) for key, value in self.normalized_mask_labels.items()]

        return image, mask, labels


# Dataloader Creation Function
def create_dataloader(
    root_dir,
    dataset_name,
    split="train",
    batch_size=8,
    shuffle=True,
    transform=None,
    mask_transform=None,
    mask_suffix="",
    mask_ext=".png",
):
    dataset = ImageMaskDataset(
        root_dir,
        dataset_name,
        split=split,
        transform=transform,
        mask_transform=mask_transform,
        mask_suffix=mask_suffix,
        mask_ext=mask_ext,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Set a fixed size for all images and masks
fixed_size = (360, 640)  # (height, width)

# Transforms for Images and Masks
transform = transforms.Compose([transforms.Resize(fixed_size), transforms.ToTensor()])

mask_transform = transforms.Compose([transforms.Resize(fixed_size), transforms.ToTensor()])

# COCO Dataset Loaders
coco_root_dir = "coco"
coco_loader_train = create_dataloader(
    coco_root_dir, "COCO", split="train", transform=transform, mask_transform=mask_transform
)

coco_loader_val = create_dataloader(
    coco_root_dir, "COCO", split="val", transform=transform, mask_transform=mask_transform
)

coco_loader_test = create_dataloader(coco_root_dir, "COCO", split="test", transform=transform)

# Pascal VOC Dataset Loaders, no test set
voc_root_dir = "voc"
voc_loader_train = create_dataloader(
    voc_root_dir, "VOC", split="train", transform=transform, mask_transform=mask_transform
)

voc_loader_val = create_dataloader(voc_root_dir, "VOC", split="val", transform=transform, mask_transform=mask_transform)

# Oxford-IIIT Pet Dataset Loaders (trainval folder for both train and val)
pets_root_dir = "oxford_pet"
pets_loader_train = create_dataloader(
    pets_root_dir, "PET", split="train", transform=transform, mask_transform=mask_transform
)

pets_loader_test = create_dataloader(pets_root_dir, "PET", split="test", transform=transform)

# MSD Brain Tumor Dataset Loaders
brain_root_dir = "msd_brain"
brain_loader_train = create_dataloader(
    brain_root_dir, "BRAIN", split="train", transform=transform, mask_transform=mask_transform
)

brain_loader_test = create_dataloader(brain_root_dir, "BRAIN", split="test", transform=transform)

# MSD Heart Dataset Loaders
heart_root_dir = "msd_heart"
heart_loader_train = create_dataloader(
    heart_root_dir, "HEART", split="train", transform=transform, mask_transform=mask_transform
)

heart_loader_test = create_dataloader(heart_root_dir, "HEART", split="test", transform=transform)

# Verifying the Dataloader Outputs
import matplotlib.pyplot as plt


def verify_dataloader(dataloader, name, num_samples=4):
    for batch in dataloader:
        if isinstance(batch, list) and len(batch) == 3:
            images, masks, labels = batch
            print(f"{name} Dataloader: Image batch shape: {images.shape}, Mask batch shape: {masks.shape}")
        else:  # Test case: only images
            images = batch
            print(f"{name} Dataloader: Image batch shape: {images.shape}")

        # Display a few samples
        fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
        for i in range(num_samples):
            img = images[i].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
            axs[i, 0].imshow(img)
            axs[i, 0].set_title("Image")
            axs[i, 0].axis("off")

            if len(batch) == 3:  # Only display masks for train/val
                mask = masks[i].permute(1, 2, 0).numpy()  # Remove channel dimension for mask
                axs[i, 1].imshow(mask)
                axs[i, 1].set_title("Mask")
                axs[i, 1].axis("off")

        plt.tight_layout()
        plt.show()
        break  # Only verify the first batch


# Verify each dataloader
verify_dataloader(coco_loader_train, "COCO Train")
verify_dataloader(coco_loader_val, "COCO Val")
verify_dataloader(coco_loader_test, "COCO Test")
verify_dataloader(voc_loader_train, "VOC Train")
verify_dataloader(voc_loader_val, "VOC Val")
verify_dataloader(pets_loader_train, "Oxford-IIIT Pets Train")
verify_dataloader(pets_loader_test, "Oxford-IIIT Pets Test")
verify_dataloader(brain_loader_train, "MSD Brain Tumor Train")
verify_dataloader(brain_loader_test, "MSD Brain Tumor Test")
verify_dataloader(heart_loader_train, "MSD Heart Train")
verify_dataloader(heart_loader_test, "MSD Heart Test")
