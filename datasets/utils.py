import json
import os
import zipfile
from typing import Dict, List

import numpy as np
import requests
from skimage.draw import polygon


def save_label_mappings(mask_dir: str, category_mapping: Dict[int, str], dataset_name: str) -> None:
    """
    Saves the category mappings as a JSON file in the specified directory.

    Args:
        mask_dir (str): Directory to save the label mapping file.
        category_mapping (Dict[int, str]): Dictionary mapping category IDs to category names.
        dataset_name (str): Name of the dataset, used to name the JSON file.

    Returns:
        None
    """
    label_mapping_path = os.path.join(mask_dir, f"{dataset_name}_label_mapping.json")
    with open(label_mapping_path, "w") as json_file:
        json.dump(category_mapping, json_file, indent=4)


def create_masks(
    image_info: Dict[str, int],
    annotations: Dict[int, List[Dict]],
    output_dir_masks: str,
    category_id_to_idx: Dict[int, int],
) -> None:
    """
    Creates binary mask files for each image, with regions filled by class indices.

    Args:
        image_info (Dict[str, int]): Dictionary containing image metadata with "id", "width", and "height".
        annotations (Dict[int, List[Dict]]): Dictionary with image IDs as keys and a list of annotation dictionaries as
        values.
        output_dir_masks (str): Directory where the generated mask files will be saved.
        category_id_to_idx (Dict[int, int]): Mapping of original category IDs to new indices.

    Returns:
        None
    """
    image_id = image_info["id"]
    width, height = image_info["width"], image_info["height"]

    # Initialize an empty mask [H, W] with zeros (background class)
    mask_array = np.zeros((height, width), dtype=np.int64)

    # Process annotations for the image
    if image_id in annotations:
        for annotation in annotations[image_id]:
            segmentation = annotation["segmentation"]
            original_category_id = annotation["category_id"]
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


def download_and_extract(url: str, zip_path: str, extract_dir: str) -> None:
    """
    Downloads a zip file from a given URL and extracts its contents into a specified directory.

    Args:
        url (str): URL to download the zip file from.
        zip_path (str): Path to save the downloaded zip file.
        extract_dir (str): Directory to extract files to.

    Returns:
        None
    """
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
