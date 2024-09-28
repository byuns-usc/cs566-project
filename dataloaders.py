import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

os.chdir('data')

# Image-Mask Dataset Class
class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, mask_suffix="", mask_ext=".png"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.mask_ext = mask_ext
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = sorted(os.listdir(self.image_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Extract the base name without extension
        base_name = os.path.splitext(img_name)[0]
        # Create the mask name by adding the suffix and using the correct extension
        mask_name = f"{base_name}{self.mask_suffix}{self.mask_ext}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Dataloader Creation Function
def create_dataloader(image_dir, mask_dir, batch_size=8, shuffle=True, transform=None, mask_transform=None, mask_suffix="", mask_ext=".png"):
    dataset = ImageMaskDataset(image_dir, mask_dir, transform=transform, mask_transform=mask_transform, mask_suffix=mask_suffix, mask_ext=mask_ext)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Set a fixed size for all images and masks
fixed_size = (360, 640)  # (height, width)

# Transforms for Images and Masks
transform = transforms.Compose([
    transforms.Resize(fixed_size),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize(fixed_size),
    transforms.ToTensor()
])

# COCO Dataset
coco_image_dir = 'coco/images'
coco_mask_dir = 'coco/masks'

coco_loader = create_dataloader(
    coco_image_dir,
    coco_mask_dir,
    transform=transform,
    mask_transform=mask_transform,
    mask_suffix="",
    mask_ext=".png"
)

# Pascal VOC Dataset
voc_image_dir = 'voc/images'
voc_mask_dir = 'voc/masks'

voc_loader = create_dataloader(
    voc_image_dir,
    voc_mask_dir,
    transform=transform,
    mask_transform=mask_transform,
    mask_suffix="",  # Mask files have the same base name without extra suffix
    mask_ext=".png"
)

# Oxford-IIIT Pet Dataset
pets_image_dir = 'oxford_pet/images'
pets_mask_dir = 'oxford_pet/masks'

pets_loader = create_dataloader(
    pets_image_dir,
    pets_mask_dir,
    transform=transform,
    mask_transform=mask_transform,
    mask_suffix="",  # Mask files have the same base name without extra suffix
    mask_ext=".png"
)

# MSD Brain Tumor Dataset
brain_image_dir = 'msd_brain/images'
brain_mask_dir = 'msd_brain/masks'

brain_loader = create_dataloader(
    brain_image_dir,
    brain_mask_dir,
    transform=transform,
    mask_transform=mask_transform,
    mask_suffix="",
    mask_ext=".png"
)

# MSD Heart Dataset
heart_image_dir = 'msd_heart/images'
heart_mask_dir = 'msd_heart/masks'

heart_loader = create_dataloader(
    heart_image_dir,
    heart_mask_dir,
    transform=transform,
    mask_transform=mask_transform,
    mask_suffix="",
    mask_ext=".png"
)

# Verifying the Dataloader Outputs
#import matplotlib.pyplot as plt

def verify_dataloader(dataloader, name, num_samples=4):
    for images, masks in dataloader:
        print(f"{name} Dataloader:")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Mask batch shape: {masks.shape}")

        # Display a few samples from the batch
        # fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
        # for i in range(num_samples):
        #     img = images[i].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        #     mask = masks[i].squeeze().numpy()  # Remove channel dimension for mask if any
            
        #     axs[i, 0].imshow(img)
        #     axs[i, 0].set_title('Image')
        #     axs[i, 0].axis('off')
            
        #     axs[i, 1].imshow(mask, cmap='gray')
        #     axs[i, 1].set_title('Mask')
        #     axs[i, 1].axis('off')
        
        # plt.tight_layout()
        # plt.show()
        break  # Only verify the first batch


# Verify each dataloader
verify_dataloader(coco_loader, "COCO")
verify_dataloader(voc_loader, "VOC")
verify_dataloader(pets_loader, "Oxford-IIIT Pets")
verify_dataloader(brain_loader, "MSD Brain Tumor")
verify_dataloader(heart_loader, "MSD Heart")
