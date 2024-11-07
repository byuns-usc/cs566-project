import json
import os
import random
import tarfile
import urllib
import zipfile

import nibabel as nib
import numpy as np
import requests
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image, ImageDraw
from skimage.draw import polygon
from sklearn.model_selection import train_test_split


# download and save images based on file name from COCO
def download_image(img_filename, output_dir_images, split):
    img_url = f"http://images.cocodataset.org/{split}2017/{img_filename}"
    img_path = os.path.join(output_dir_images, img_filename)
    img_data = requests.get(img_url).content
    with open(img_path, "wb") as f:
        f.write(img_data)


# Save label mappings to a JSON file
def save_label_mappings(mask_dir, category_mapping, dataset_name):
    label_mapping_path = os.path.join(mask_dir, f"{dataset_name}_label_mapping.json")
    with open(label_mapping_path, "w") as json_file:
        json.dump(category_mapping, json_file, indent=4)


# Create mask files as [H, W] arrays with class indices
def create_masks(image_info, annotations, output_dir_masks, category_id_to_idx):
    image_id = image_info["id"]
    width, height = image_info["width"], image_info["height"]

    # Initialize an empty mask [H, W] with zeros (background class)
    mask_array = np.zeros((height, width), dtype=np.int64)

    # Process annotations for the image
    if image_id in annotations:
        for annotation in annotations[image_id]:
            segmentation = annotation["segmentation"]
            original_category_id = annotation["category_id"]  # Use category_id as class index
            category_id = int(category_id_to_idx[original_category_id])
            # If the segmentation is a list of polygons
            if isinstance(segmentation, list):
                for polygon_points in segmentation:
                    # Convert polygon points to (x, y) coordinates
                    poly = np.array(polygon_points).reshape(-1, 2)

                    # Extract row (y) and column (x) coordinates for the polygon
                    rr, cc = polygon(poly[:, 1], poly[:, 0], mask_array.shape)

                    # Ensure the coordinates are within the bounds of the mask
                    rr = np.clip(rr, 0, mask_array.shape[0] - 1)
                    cc = np.clip(cc, 0, mask_array.shape[1] - 1)

                    # Fill the mask with the category_id
                    mask_array[rr, cc] = category_id
    # Save the mask as a .npy file
    mask_path = os.path.join(output_dir_masks, f"{image_id:012d}.npy")
    np.save(mask_path, mask_array)


# process a single COCO split
def process_coco_split(split, category_id_to_idx):
    output_dir_images = f"coco/{split}/images"
    output_dir_masks = f"coco/{split}/masks"

    if split != "test":
        os.makedirs(output_dir_masks, exist_ok=True)

    os.makedirs(output_dir_images, exist_ok=True)

    if split != "test":
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        annotations_zip_path = "annotations.zip"
        annotations_dir = "annotations"

        # Download and extract annotations if not already present
        if not os.path.exists(annotations_zip_path):
            print("Downloading annotations...")
            annotations_data = requests.get(annotations_url)
            with open(annotations_zip_path, "wb") as f:
                f.write(annotations_data.content)
            print("Annotations downloaded.")
            with zipfile.ZipFile(annotations_zip_path, "r") as zip_ref:
                zip_ref.extractall(annotations_dir)

        # Load the appropriate annotation file for train or val
        annotations_file = os.path.join(annotations_dir, "annotations", f"instances_{split}2017.json")
        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        # Get all image data entries
        images_data = coco_data["images"]

        # Create a mapping from image_id to annotation data
        image_id_to_annotations = {}
        for annotation in coco_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in image_id_to_annotations:
                image_id_to_annotations[image_id] = []
            image_id_to_annotations[image_id].append(annotation)

        # Download images and create masks
        for image_info in images_data:
            img_filename = image_info["file_name"]
            download_image(img_filename, output_dir_images, split)
            create_masks(image_info, image_id_to_annotations, output_dir_masks, category_id_to_idx)

    # For test split, only download images
    else:
        print(f"Processing {split} split.")
        test_images_url = f"http://images.cocodataset.org/zips/{split}2017.zip"
        test_zip_path = f"{split}2017.zip"

        # Download test images
        download_and_extract(test_images_url, test_zip_path, output_dir_images)
        print(f"Downloaded test images for {split} split.")


# Utility to download and extract large zip files in chunks
def download_and_extract(url, zip_path, extract_dir):
    if not os.path.exists(zip_path):
        print(f"Downloading from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Extracting {zip_path} to {extract_dir}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Manually extract files and place them directly in the images folder
            for member in zip_ref.namelist():
                # We are specifically looking for files under the 'test2017/' subfolder
                if member.startswith("test2017/") and not member.endswith("/"):
                    # Strip the 'test2017/' part of the path
                    filename = os.path.basename(member)
                    # Define where to place the extracted files
                    extracted_path = os.path.join(extract_dir, filename)

                    # Open the source file from the zip and write it to the destination
                    with zip_ref.open(member) as source_file:
                        with open(extracted_path, "wb") as output_file:
                            output_file.write(source_file.read())
        print(f"Successfully extracted files to {extract_dir} without 'test2017' subfolder.")
    else:
        print(f"{zip_path} already exists.")


def COCO():
    # Load annotations once for both train and val splits to ensure consistent colors and save category labels
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_zip_path = "annotations.zip"
    annotations_dir = "annotations"

    if not os.path.exists(annotations_zip_path):
        print("Downloading annotations...")
        annotations_data = requests.get(annotations_url)
        with open(annotations_zip_path, "wb") as f:
            f.write(annotations_data.content)
        print("Annotations downloaded.")
        with zipfile.ZipFile(annotations_zip_path, "r") as zip_ref:
            zip_ref.extractall(annotations_dir)

    # Load the annotations to create a global category-to-color mapping
    annotations_file = os.path.join(annotations_dir, "annotations", "instances_train2017.json")
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    original_label_mapping = {category["name"]: category["id"] for category in coco_data["categories"]}
    category_id_to_idx = {}
    label_mapping = {}
    for idx, (name, id) in enumerate(original_label_mapping.items(), start=1):
        category_id_to_idx[id] = idx
        label_mapping[name] = idx
    label_mapping["background"] = 0
    category_id_to_idx[0] = 0
    # Ensure the same mapping is used for both train and val
    process_coco_split("train", category_id_to_idx)
    process_coco_split("val", category_id_to_idx)
    process_coco_split("test", category_id_to_idx)

    # Save the label mappings with category names and colors
    save_label_mappings("coco/train/masks", label_mapping, "Train")
    save_label_mappings("coco/val/masks", label_mapping, "Val")


# Process a single VOC split and save images/masks
def process_voc_split(voc_dataset, output_dir_images, output_dir_masks, split_name):
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)

    for idx in range(len(voc_dataset)):
        # Get the image and mask pair
        img, mask = voc_dataset[idx]

        # Save the image
        img = T.ToPILImage()(img)
        img_path = os.path.join(output_dir_images, f"{split_name}_{idx}.jpg")
        img.save(img_path)

        # Convert the mask to a NumPy array with shape [H, W]
        mask_array = np.array(mask)
        mask_array[mask_array == 255] = 0

        # Save the mask as a .npy file
        mask_path = os.path.join(output_dir_masks, f"{split_name}_{idx}.npy")
        np.save(mask_path, mask_array)
    print(f"{split_name.capitalize()} PASCAL VOC images and masks have been saved!")


def VOC():
    # Output directories for images and masks for all sets
    base_dir = "voc"
    os.makedirs(base_dir, exist_ok=True)

    class_names = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    # Assign a random color to each class
    label_mapping = {class_names[i]: i for i in range(len(class_names))}

    # Process Train Set
    train_output_dir_images = os.path.join(base_dir, "train/images")
    train_output_dir_masks = os.path.join(base_dir, "train/masks")
    train_dataset = datasets.VOCSegmentation(
        root="./data", year="2012", image_set="train", download=True, transform=T.ToTensor()
    )
    process_voc_split(train_dataset, train_output_dir_images, train_output_dir_masks, "train")

    # Process Val Set
    val_output_dir_images = os.path.join(base_dir, "val/images")
    val_output_dir_masks = os.path.join(base_dir, "val/masks")
    val_dataset = datasets.VOCSegmentation(
        root="./data", year="2012", image_set="val", download=True, transform=T.ToTensor()
    )
    process_voc_split(val_dataset, val_output_dir_images, val_output_dir_masks, "val")

    save_label_mappings("voc/train/masks", label_mapping, "Train")
    save_label_mappings("voc/val/masks", label_mapping, "Val")


def get_class_names():
    # List of breed names from the Oxford-IIIT Pet Dataset
    return [
        "Abyssinian",
        "Bengal",
        "Birman",
        "Bombay",
        "British_Shorthair",
        "Egyptian_Mau",
        "Maine_Coon",
        "Persian",
        "Ragdoll",
        "Russian_Blue",
        "Siamese",
        "Sphynx",
        "American_Bulldog",
        "American_Pit_Bull_Terrier",
        "Basset_Hound",
        "Beagle",
        "Boxer",
        "Chihuahua",
        "English_Cocker_Spaniel",
        "English_Setter",
        "German_Shorthaired",
        "Great_Pyrenees",
        "Havanese",
        "Japanese_Chin",
        "Keeshond",
        "Leonberger",
        "Miniature_Pinscher",
        "Newfoundland",
        "Pomeranian",
        "Pug",
        "Saint_Bernard",
        "Samoyed",
        "Scottish_Terrier",
        "Shiba_Inu",
        "Staffordshire_Bull_Terrier",
        "Wheaten_Terrier",
        "Yorkshire_Terrier",
    ]


def process_pet_split(pet_dataset, class_mapping, output_dir_images, output_dir_masks, split_name, indices):
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)

    for idx, dataset_idx in enumerate(indices):
        # Get the image and mask
        img, (category, mask) = pet_dataset[dataset_idx]

        # Save the image
        img_path = os.path.join(output_dir_images, f"{split_name}_{idx}.jpg")
        img.save(img_path)

        # Convert mask to NumPy and map foreground to breed ID
        mask_array = np.array(mask)
        mask_array = np.where(mask_array == 2, 0, category + 1)

        # Save mask as .npy
        mask_path = os.path.join(output_dir_masks, f"{split_name}_{idx}.npy")
        np.save(mask_path, mask_array)
    print(f"{split_name.capitalize()} set!")


def split_trainval_indices(dataset_size):
    indices = np.arange(dataset_size)
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    return train_indices, val_indices, test_indices


# Main function to handle the Oxford-IIIT Pet dataset
def PET():
    base_dir = "oxford_pet"
    os.makedirs(base_dir, exist_ok=True)

    # Get actual breed names from the dataset
    class_names = ["Background"] + get_class_names()

    # Create class name-to-ID mapping
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    # Load trainval dataset
    pet_trainval_dataset = datasets.OxfordIIITPet(
        root="./data", split="trainval", target_types=("category", "segmentation"), download=True
    )

    # Split trainval into train, val, and test sets
    train_indices, val_indices, test_indices = split_trainval_indices(len(pet_trainval_dataset))

    # Process each split
    process_pet_split(
        pet_trainval_dataset,
        class_mapping,
        os.path.join(base_dir, "train/images"),
        os.path.join(base_dir, "train/masks"),
        "train",
        train_indices,
    )

    process_pet_split(
        pet_trainval_dataset,
        class_mapping,
        os.path.join(base_dir, "val/images"),
        os.path.join(base_dir, "val/masks"),
        "val",
        val_indices,
    )

    process_pet_split(
        pet_trainval_dataset,
        class_mapping,
        os.path.join(base_dir, "test/images"),
        os.path.join(base_dir, "test/masks"),
        "test",
        test_indices,
    )
    save_label_mappings("oxford_pet/train/masks", class_mapping, "Train")
    save_label_mappings("oxford_pet/val/masks", class_mapping, "Val")
    save_label_mappings("oxford_pet/test/masks", class_mapping, "Test")


def process_pet2_split(pet_dataset, output_dir_images, output_dir_masks, split_name, indices):
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)

    for idx, dataset_idx in enumerate(indices):
        # Get the image and mask
        img, (category, mask) = pet_dataset[dataset_idx]

        # Save the image
        img_path = os.path.join(output_dir_images, f"{split_name}_{idx}.jpg")
        img.save(img_path)

        # Convert mask to NumPy and map foreground to breed ID
        mask_array = np.array(mask)
        mask_array = np.where(mask_array > 0, mask_array - 1, 1)  # 2 is background, 1 pet, 3 foreground
        # Save mask as .npy
        mask_path = os.path.join(output_dir_masks, f"{split_name}_{idx}.npy")
        np.save(mask_path, mask_array)
    print(f"{split_name.capitalize()} set!")


# Main function to handle the Oxford-IIIT Pet dataset
def PET2():
    base_dir = "oxford_pet2"
    os.makedirs(base_dir, exist_ok=True)

    # Load trainval dataset
    pet_trainval_dataset = datasets.OxfordIIITPet(
        root="./data", split="trainval", target_types=("category", "segmentation"), download=True
    )

    # Split trainval into train, val, and test sets
    train_indices, val_indices, test_indices = split_trainval_indices(len(pet_trainval_dataset))

    # Process each split
    process_pet2_split(
        pet_trainval_dataset,
        os.path.join(base_dir, "train/images"),
        os.path.join(base_dir, "train/masks"),
        "train",
        train_indices,
    )

    process_pet2_split(
        pet_trainval_dataset,
        os.path.join(base_dir, "val/images"),
        os.path.join(base_dir, "val/masks"),
        "val",
        val_indices,
    )

    process_pet2_split(
        pet_trainval_dataset,
        os.path.join(base_dir, "test/images"),
        os.path.join(base_dir, "test/masks"),
        "test",
        test_indices,
    )
    class_mapping = {"pet": 0, "background": 1, "foreground": 2}
    save_label_mappings("oxford_pet2/train/masks", class_mapping, "Train")
    save_label_mappings("oxford_pet2/val/masks", class_mapping, "Val")
    save_label_mappings("oxford_pet2/test/masks", class_mapping, "Test")


def process_heart_train(image_paths, mask_paths, output_dir_train_images, output_dir_train_masks):
    os.makedirs(output_dir_train_images, exist_ok=True)
    os.makedirs(output_dir_train_masks, exist_ok=True)

    # Process each image and mask pair
    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load the image and mask
        img_nifti = nib.load(img_path)
        mask_nifti = nib.load(mask_path)

        img_data = img_nifti.get_fdata()
        mask_data = mask_nifti.get_fdata()

        # Extract the middle 50% slices
        lower_bound = int(img_data.shape[2] * 0.25)
        upper_bound = int(img_data.shape[2] * 0.75)

        for slice_idx in range(lower_bound, upper_bound):
            img_slice = img_data[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx]

            # Normalize image slice and convert to PIL
            img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
            img_normalized = img_normalized.astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)

            # Save the image and corresponding mask in the respective directories
            img_pil.save(os.path.join(output_dir_train_images, f"train_{idx}_{slice_idx}.jpg"))
            mask_path = os.path.join(output_dir_train_masks, f"train_{idx}_{slice_idx}.npy")
            np.save(mask_path, mask_slice)

    print("MSD Heart train MRI images and masks (mid 50% slices) have been saved!")


# Function to process the MSD Heart test set (use all slices, includes masks if available)
def process_heart_test(image_paths, output_dir_test_images):
    os.makedirs(output_dir_test_images, exist_ok=True)

    # Process each image and mask pair
    for idx, img_path in enumerate(image_paths):
        # Load the image and mask
        img_nifti = nib.load(img_path)

        img_data = img_nifti.get_fdata()

        # Process all slices
        for slice_idx in range(img_data.shape[2]):
            img_slice = img_data[:, :, slice_idx]

            # Normalize image slice and convert to PIL
            img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
            img_normalized = img_normalized.astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)

            # Save the image and corresponding mask in the respective directories
            img_pil.save(os.path.join(output_dir_test_images, f"test_{idx}_{slice_idx}.jpg"))

    print("MSD Heart test MRI images (all slices) have been saved!")


# Main function to handle the MSD Heart dataset (train + test)
def HEART():
    # Output directories for train and test data
    base_dir = "msd_heart"
    output_dir_train_images = os.path.join(base_dir, "train", "images")
    output_dir_train_masks = os.path.join(base_dir, "train", "masks")
    output_dir_test_images = os.path.join(base_dir, "test", "images")
    output_dir_nifti = os.path.join(base_dir, "nifti")

    os.makedirs(output_dir_train_images, exist_ok=True)
    os.makedirs(output_dir_train_masks, exist_ok=True)
    os.makedirs(output_dir_test_images, exist_ok=True)
    os.makedirs(output_dir_nifti, exist_ok=True)

    # MSD Heart Segmentation dataset URL
    msd_heart_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar"
    msd_tar_path = "Task02_Heart.tar"

    # Download the dataset if not already present
    if not os.path.exists(msd_tar_path):
        print("Downloading MSD Heart dataset...")
        with requests.get(msd_heart_url, stream=True) as response:
            response.raise_for_status()
            with open(msd_tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    # Extract the tar file using tarfile module
    print("Extracting the dataset...")
    with tarfile.open(msd_tar_path, "r") as tar:
        tar.extractall(output_dir_nifti)

    # Directory paths for images and masks within the extracted data
    image_dir = os.path.join(output_dir_nifti, "Task02_Heart", "imagesTr")
    mask_dir = os.path.join(output_dir_nifti, "Task02_Heart", "labelsTr")

    # Filter out hidden files and other system files
    image_paths = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii.gz") and not f.startswith("._")]
    )
    mask_paths = sorted(
        [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii.gz") and not f.startswith("._")]
    )

    # Process the train set (mid 50% slices)
    process_heart_train(image_paths, mask_paths, output_dir_train_images, output_dir_train_masks)

    image_dir_test = os.path.join(output_dir_nifti, "Task02_Heart", "imagesTs")

    image_paths_test = sorted(
        [
            os.path.join(image_dir_test, f)
            for f in os.listdir(image_dir_test)
            if f.endswith(".nii.gz") and not f.startswith("._")
        ]
    )

    # Process the test set (all slices, includes masks)
    process_heart_test(image_paths_test, output_dir_test_images)

    # Store label-to-mask-value mapping for MSD Heart
    # Scale mask values (0-3) to (0-255)
    MASK_LABELS_HEART = {"background": 0, "left ventricle": 1, "myocardium": 2, "right ventricle": 3}

    # Save the label mapping as a JSON file inside the train/masks folder
    save_label_mappings(output_dir_train_masks, MASK_LABELS_HEART, "Train")
    print("MSD Heart labels.")


def process_brain_train(image_paths, mask_paths, output_dir_train_images, output_dir_train_masks):
    os.makedirs(output_dir_train_images, exist_ok=True)
    os.makedirs(output_dir_train_masks, exist_ok=True)

    # Process each image and mask pair
    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load the image and mask
        img_nifti = nib.load(img_path)
        mask_nifti = nib.load(mask_path)

        img_data = img_nifti.get_fdata()
        mask_data = mask_nifti.get_fdata()

        # Calculate the middle 50% range of slices
        num_slices = img_data.shape[2]
        lower_bound = int(num_slices * 0.25)
        upper_bound = int(num_slices * 0.75)

        # Process only the middle 50% of slices
        for slice_idx in range(lower_bound, upper_bound):
            img_slice = img_data[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx]

            # Normalize image slice and convert to PIL
            img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
            img_normalized = img_normalized.astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)

            # Save the image and corresponding mask in the train directory
            img_pil.save(os.path.join(output_dir_train_images, f"train_{idx}_{slice_idx}.png"))
            # Save the mask directly as .npy
            mask_path = os.path.join(output_dir_train_masks, f"train_{idx}_{slice_idx}.npy")
            np.save(mask_path, mask_slice)
    print("MSD Brain train MRI images and masks (mid 50% slices) have been saved!")


# Function to process the MSD Brain test set (images only)
def process_brain_test(image_paths, output_dir_test_images):
    os.makedirs(output_dir_test_images, exist_ok=True)

    # Process each image (no masks for test set)
    for idx, img_path in enumerate(image_paths):
        # Load the image
        img_nifti = nib.load(img_path)
        img_data = img_nifti.get_fdata()

        # Process all slices (only images)
        num_slices = img_data.shape[2]
        for slice_idx in range(num_slices):
            img_slice = img_data[:, :, slice_idx]

            # Normalize image slice and convert to PIL
            img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
            img_normalized = img_normalized.astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)

            # Save the image
            img_pil.save(os.path.join(output_dir_test_images, f"test_{idx}_{slice_idx}.png"))

    print("MSD Brain test MRI images have been saved!")


# Main function to handle the MSD Brain dataset (train + test)
def BRAIN():
    # Output directories for train and test data
    base_dir = "msd_brain"
    output_dir_train_images = os.path.join(base_dir, "train", "images")
    output_dir_train_masks = os.path.join(base_dir, "train", "masks")
    output_dir_test_images = os.path.join(base_dir, "test", "images")
    output_dir_nifti = os.path.join(base_dir, "nifti")

    os.makedirs(output_dir_train_images, exist_ok=True)
    os.makedirs(output_dir_train_masks, exist_ok=True)
    os.makedirs(output_dir_test_images, exist_ok=True)
    os.makedirs(output_dir_nifti, exist_ok=True)

    # MSD Brain Tumor Segmentation dataset URL
    msd_brain_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    msd_tar_path = "Task01_BrainTumour.tar"

    # Download the dataset if not already present
    if not os.path.exists(msd_tar_path):
        print("Downloading MSD Brain Tumor dataset...")
        with requests.get(msd_brain_url, stream=True) as response:
            response.raise_for_status()
            with open(msd_tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    # Extract the tar file using tarfile module
    print("Extracting the dataset...")
    with tarfile.open(msd_tar_path, "r") as tar:
        tar.extractall(output_dir_nifti)

    # Directory paths for images and masks within the extracted data
    image_dir_train = os.path.join(output_dir_nifti, "Task01_BrainTumour", "imagesTr")
    mask_dir_train = os.path.join(output_dir_nifti, "Task01_BrainTumour", "labelsTr")
    image_dir_test = os.path.join(output_dir_nifti, "Task01_BrainTumour", "imagesTs")

    # Filter out hidden files and other system files
    image_paths_train = sorted(
        [
            os.path.join(image_dir_train, f)
            for f in os.listdir(image_dir_train)
            if f.endswith(".nii.gz") and not f.startswith("._")
        ]
    )
    mask_paths_train = sorted(
        [
            os.path.join(mask_dir_train, f)
            for f in os.listdir(mask_dir_train)
            if f.endswith(".nii.gz") and not f.startswith("._")
        ]
    )
    image_paths_test = sorted(
        [
            os.path.join(image_dir_test, f)
            for f in os.listdir(image_dir_test)
            if f.endswith(".nii.gz") and not f.startswith("._")
        ]
    )

    # Process the train set (all slices)
    process_brain_train(image_paths_train, mask_paths_train, output_dir_train_images, output_dir_train_masks)

    # Process the test set (images only)
    process_brain_test(image_paths_test, output_dir_test_images)

    # Store label-to-mask-value mapping for MSD Brain
    MASK_LABELS_BRAIN = {"background": 0, "edema": 1, "non-enhancing tumor": 2, "enhancing tumor": 3}

    # Save the label mapping as a JSON file inside the train/masks folder
    save_label_mappings(output_dir_train_masks, MASK_LABELS_BRAIN, "Train")
    print("MSD Brain labels.")

def process_lung_train(image_paths, mask_paths, output_dir_train_images, output_dir_train_masks):
    os.makedirs(output_dir_train_images, exist_ok=True)
    os.makedirs(output_dir_train_masks, exist_ok=True)

    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        img_nifti = nib.load(img_path)
        mask_nifti = nib.load(mask_path)

        img_data = img_nifti.get_fdata()
        mask_data = mask_nifti.get_fdata()

        lower_bound = int(img_data.shape[2] * 0.35)
        upper_bound = int(img_data.shape[2] * 0.85)

        for slice_idx in range(lower_bound, upper_bound):
            img_slice = img_data[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx]

            img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
            img_normalized = img_normalized.astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)

            img_pil.save(os.path.join(output_dir_train_images, f"train_{idx}_{slice_idx}.jpg"))
            mask_path = os.path.join(output_dir_train_masks, f"train_{idx}_{slice_idx}.npy")
            np.save(mask_path, mask_slice)
            print(np.unique(mask_slice))

    print("MSD Lung train CT images and masks (mid 50% slices) have been saved!")

def process_lung_test(image_paths, output_dir_test_images):
    os.makedirs(output_dir_test_images, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        img_nifti = nib.load(img_path)
        img_data = img_nifti.get_fdata()

        for slice_idx in range(img_data.shape[2]):
            img_slice = img_data[:, :, slice_idx]

            img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
            img_normalized = img_normalized.astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)

            img_pil.save(os.path.join(output_dir_test_images, f"test_{idx}_{slice_idx}.jpg"))

    print("MSD Lung test CT images (all slices) have been saved!")

def LUNG():
    base_dir = "msd_lung"
    output_dir_train_images = os.path.join(base_dir, "train", "images")
    output_dir_train_masks = os.path.join(base_dir, "train", "masks")
    output_dir_test_images = os.path.join(base_dir, "test", "images")
    output_dir_nifti = os.path.join(base_dir, "nifti")

    os.makedirs(output_dir_train_images, exist_ok=True)
    os.makedirs(output_dir_train_masks, exist_ok=True)
    os.makedirs(output_dir_test_images, exist_ok=True)
    os.makedirs(output_dir_nifti, exist_ok=True)

    msd_lung_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar"
    msd_tar_path = "Task06_Lung.tar"

    if not os.path.exists(msd_tar_path):
        print("Downloading MSD Lung dataset...")
        with requests.get(msd_lung_url, stream=True) as response:
            response.raise_for_status()
            with open(msd_tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    print("Extracting the dataset...")
    with tarfile.open(msd_tar_path, "r") as tar:
        tar.extractall(output_dir_nifti)

    image_dir = os.path.join(output_dir_nifti, "Task06_Lung", "imagesTr")
    mask_dir = os.path.join(output_dir_nifti, "Task06_Lung", "labelsTr")

    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii.gz") and not f.startswith("._")])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii.gz") and not f.startswith("._")])

    process_lung_train(image_paths, mask_paths, output_dir_train_images, output_dir_train_masks)

    image_dir_test = os.path.join(output_dir_nifti, "Task06_Lung", "imagesTs")
    image_paths_test = sorted([os.path.join(image_dir_test, f) for f in os.listdir(image_dir_test) if f.endswith(".nii.gz") and not f.startswith("._")])

    process_lung_test(image_paths_test, output_dir_test_images)

    MASK_LABELS_LUNG = {"background": 0, "lung tumor": 1}
    save_label_mappings(output_dir_train_masks, MASK_LABELS_LUNG, "Train")



def main():
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    # COCO()
    # VOC()
    # PET()
    PET2()
    # HEART()
    # BRAIN()
    LUNG()


if __name__ == "__main__":
    main()
