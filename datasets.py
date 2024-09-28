import os
import requests
import json
import numpy as np
from PIL import Image, ImageDraw
import random
from torchvision import datasets
import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
import tarfile
import nibabel as nib

os.makedirs('data', exist_ok=True)
os.chdir('data')

####COCO####
# Output directories for images and masks
output_dir_images = 'coco/images'
output_dir_masks = 'coco/masks'

# Create output directories if they don't exist
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)

# Download the COCO annotations file
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
annotations_zip_path = 'annotations.zip'
annotations_dir = 'annotations'

# Download and extract annotations if not already present
if not os.path.exists(annotations_zip_path):
    print("Downloading annotations...")
    annotations_data = requests.get(annotations_url)
    with open(annotations_zip_path, 'wb') as f:
        f.write(annotations_data.content)
    print("Annotations downloaded.")

    import zipfile
    with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
        zip_ref.extractall(annotations_dir)

# Set the path to the extracted annotations file
annotations_file = os.path.join(annotations_dir, 'annotations', 'instances_train2017.json')

# Load the annotations file
with open(annotations_file, 'r') as f:
    coco_data = json.load(f)

# Get the first 10 image data entries
##### CHANGE THIS WHEN TRAINING
images_data = coco_data['images'][:10]

# Create a mapping from image_id to annotation data
image_id_to_annotations = {}
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    if image_id not in image_id_to_annotations:
        image_id_to_annotations[image_id] = []
    image_id_to_annotations[image_id].append(annotation)

# Create a mapping from category ID to a random color
category_id_to_color = {}
categories = coco_data['categories']
for category in categories:
    category_id = category['id']
    # Assign a random RGB color to each category
    category_id_to_color[category_id] = tuple([random.randint(0, 255) for _ in range(3)])

# Download images and create colored masks
for image_info in images_data:
    image_id = image_info['id']
    img_filename = image_info['file_name']
    img_height = image_info['height']
    img_width = image_info['width']
    
    # Download the image and save it
    img_url = f"http://images.cocodataset.org/train2017/{img_filename}"
    img_path = os.path.join(output_dir_images, img_filename)
    img_data = requests.get(img_url).content
    with open(img_path, 'wb') as f:
        f.write(img_data)
    
    # Create a blank RGB mask
    mask = Image.new('RGB', (img_width, img_height), (0, 0, 0))
    
    # Draw the polygons on the mask
    annotations = image_id_to_annotations.get(image_id, [])
    for annotation in annotations:
        segmentation = annotation['segmentation']
        category_id = annotation['category_id']
        color = category_id_to_color[category_id]  # Get color for the category
        
        if isinstance(segmentation, list):  # Ensure it's a list of polygons
            for polygon in segmentation:
                # Convert the polygon points into a format suitable for drawing
                poly = np.array(polygon).reshape((len(polygon) // 2, 2))
                # Draw the polygon as a filled area with its category color
                ImageDraw.Draw(mask).polygon(poly.flatten().tolist(), outline=color, fill=color)
    
    # Save the mask as a PNG file
    mask_path = os.path.join(output_dir_masks, f"{os.path.splitext(img_filename)[0]}.png")
    mask.save(mask_path)

print("Images and colored masks have been downloaded and saved!")

#####VOC####

# Output directories for images and masks
output_dir_images = 'voc/images'
output_dir_masks = 'voc/masks'

# Create output directories if they don't exist
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)

# Download Pascal VOC dataset (train set) - selecting only 10 samples
voc_dataset = datasets.VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=T.ToTensor())

# Pascal VOC has 21 classes (including background)
class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Assign a random color to each class
class_to_color = {cls_id: tuple([random.randint(0, 255) for _ in range(3)]) for cls_id in range(len(class_names))}

# Process the first 10 samples only
#### CHANGE THIS WHILE TRAINING
# len(voc_dataset)
for idx in range(10):
    # Get the image and mask pair
    img, mask = voc_dataset[idx]
    
    # Save the image
    img = T.ToPILImage()(img)
    img_path = os.path.join(output_dir_images, f'{idx}.jpg')
    img.save(img_path)
    
    # Prepare mask for coloring
    mask_array = np.array(mask)
    colored_mask = Image.new('RGB', mask_array.shape[::-1], (0, 0, 0))
    
    # Draw colors for each class
    for class_id, color in class_to_color.items():
        # Create a binary mask for the class and apply the color
        class_mask = (mask_array == class_id)
        ImageDraw.Draw(colored_mask).bitmap((0, 0), Image.fromarray(class_mask.astype('uint8') * 255), fill=color)
    
    # Save the colored mask
    mask_path = os.path.join(output_dir_masks, f'{idx}.png')
    colored_mask.save(mask_path)

print("Pascal VOC images and colored masks have been saved!")

####OXFORD PET###


# Output directories for images and masks
output_dir_images = 'oxford_pet/images'
output_dir_masks = 'oxford_pet/masks'

# Create output directories if they don't exist
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)

# Download the Oxford-IIIT Pet Dataset (only 10 samples for testing)
oxford_pet_dataset = datasets.OxfordIIITPet(
    root='./data',
    split='trainval',  # train + validation split
    target_types='segmentation',
    download=True
)

# Assign random colors for visualization of masks
class_to_color = {cls_id: tuple([random.randint(0, 255) for _ in range(3)]) for cls_id in range(3)}

# Process only 10 samples
#### CHANGE - len(oxford_pet_dataset)
for idx in range(10):
    # Get the image and segmentation mask pair
    img, mask = oxford_pet_dataset[idx]
    
    # Save the image
    img_path = os.path.join(output_dir_images, f'{idx}.jpg')
    img.save(img_path)
    
    # Prepare mask for coloring
    mask_array = np.array(mask)
    colored_mask = Image.new('RGB', mask_array.shape[::-1], (0, 0, 0))
    
    # Draw colors for each class
    for class_id, color in class_to_color.items():
        # Create a binary mask for the class and apply the color
        class_mask = (mask_array == class_id)
        ImageDraw.Draw(colored_mask).bitmap((0, 0), Image.fromarray(class_mask.astype('uint8') * 255), fill=color)
    
    # Save the colored mask
    mask_path = os.path.join(output_dir_masks, f'{idx}.png')
    colored_mask.save(mask_path)

print("Oxford-IIIT Pet images and colored masks have been saved!")

#####HEART#####


# Output directories for images and masks
output_dir_images = 'msd_heart/images'
output_dir_masks = 'msd_heart/masks'
output_dir_nifti = 'msd_heart/nifti'

# Create output directories if they don't exist
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)
os.makedirs(output_dir_nifti, exist_ok=True)

# MSD Heart Segmentation dataset URL
msd_heart_url = 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar'

# Download path for the dataset
msd_tar_path = 'Task02_Heart.tar'

# Download the dataset if not already present
if not os.path.exists(msd_tar_path):
    print("Downloading MSD Heart dataset...")
    with requests.get(msd_heart_url, stream=True) as response:
        response.raise_for_status()
        with open(msd_tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Extract the tar file using tarfile module
print("Extracting the dataset...")
with tarfile.open(msd_tar_path, 'r') as tar:
    tar.extractall(output_dir_nifti)

# Directory paths for images and masks within the extracted data
image_dir = os.path.join(output_dir_nifti, 'Task02_Heart', 'imagesTr')
mask_dir = os.path.join(output_dir_nifti, 'Task02_Heart', 'labelsTr')

# Filter valid .nii.gz files only (exclude hidden files starting with "._")
### CHANGE remove [:1]
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz') and not f.startswith('._')])[:1]
mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz') and not f.startswith('._')])[:1]

# Process and save the first 10 samples
for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
    # Load the image and mask
    img_nifti = nib.load(img_path)
    mask_nifti = nib.load(mask_path)
    
    img_data = img_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()
    
    for slice_idx in range(img_data.shape[2]):
        # Extract the middle slice from 3D volume
        img_slice = img_data[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx]
        
        # Normalize image slice and convert to PIL
        img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
        img_normalized = img_normalized.astype(np.uint8)
        img_pil = Image.fromarray(img_normalized)
        
        # Save the image
        img_pil.save(os.path.join(output_dir_images, f'{idx}_{slice_idx}.jpg'))
        
        # Save the mask
        mask_pil = Image.fromarray(mask_slice.astype(np.uint8) * 255)
        mask_pil.save(os.path.join(output_dir_masks, f'{idx}_{slice_idx}.png'))

print("MSD Heart MRI images and masks have been saved!")

######BRAIN######

# Output directories for images and masks
output_dir_images = 'msd_brain/images'
output_dir_masks = 'msd_brain/masks'
output_dir_nifti = 'msd_brain/nifti'

# Create output directories if they don't exist
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)
os.makedirs(output_dir_nifti, exist_ok=True)

# MSD Brain Tumor Segmentation dataset URL (Sample for testing)
msd_brain_url = 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar'

# Download path for the dataset
msd_tar_path = 'Task01_BrainTumour.tar'

# Download the dataset if not already present
if not os.path.exists(msd_tar_path):
    print("Downloading MSD Brain Tumor dataset...")
    with requests.get(msd_brain_url, stream=True) as response:
        response.raise_for_status()
        with open(msd_tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Extract the tar file using tarfile module
print("Extracting the dataset...")
with tarfile.open(msd_tar_path, 'r') as tar:
    tar.extractall(output_dir_nifti)

# Directory paths for images and masks within the extracted data
image_dir = os.path.join(output_dir_nifti, 'Task01_BrainTumour', 'imagesTr')
mask_dir = os.path.join(output_dir_nifti, 'Task01_BrainTumour', 'labelsTr')

# Filter valid .nii.gz files only (exclude hidden files starting with "._")
# CHANGE
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz') and not f.startswith('._')])[:1]
mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz') and not f.startswith('._')])[:1]

# Process and save all slices 
for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
    # Load the image and mask
    img_nifti = nib.load(img_path)
    mask_nifti = nib.load(mask_path)
    
    img_data = img_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()
    
    # Iterate through all slices in the volume
    num_slices = img_data.shape[2]
    for slice_idx in range(num_slices):
        # Extract slice
        img_slice = img_data[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx]
        
        # Normalize image slice and convert to uint8 for visualization
        img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
        img_normalized = img_normalized.astype(np.uint8)
        img_pil = Image.fromarray(img_normalized)
        
        # Save the image
        img_pil.save(os.path.join(output_dir_images, f'{idx}_{slice_idx}.png'))
        
        # Ensure the mask is properly scaled and converted to uint8
        if mask_slice.max() > 0:
            mask_enhanced = (mask_slice * (255 / mask_slice.max())).astype(np.uint8)
        else:
            mask_enhanced = mask_slice.astype(np.uint8)
        
        # Save the mask
        mask_pil = Image.fromarray(mask_enhanced)
        mask_pil.save(os.path.join(output_dir_masks, f'{idx}_{slice_idx}.png'))

print("All slices have been saved!")

