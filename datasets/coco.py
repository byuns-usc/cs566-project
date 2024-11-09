import json
import os
import zipfile

import requests

from .base_dataset import BaseDataset
from .utils import create_masks, download_and_extract, save_label_mappings


class COCODataset(BaseDataset):
    def __init__(self, base_dir):
        super().__init__(os.path.join(base_dir, "coco"))
        self.annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        self.annotations_zip_path = "annotations.zip"
        self.annotations_dir = "annotations"

    def download_image(self, img_filename, output_dir_images, split):
        """
        Download and save an image from COCO dataset.
        """
        img_url = f"http://images.cocodataset.org/{split}2017/{img_filename}"
        img_path = os.path.join(output_dir_images, img_filename)
        img_data = requests.get(img_url).content
        with open(img_path, "wb") as f:
            f.write(img_data)

    def download(self):
        if not os.path.exists(self.annotations_zip_path):
            print("Downloading annotations...")
            annotations_data = requests.get(self.annotations_url)
            with open(self.annotations_zip_path, "wb") as f:
                f.write(annotations_data.content)
            print("Annotations downloaded.")

        with zipfile.ZipFile(self.annotations_zip_path, "r") as zip_ref:
            zip_ref.extractall(self.annotations_dir)

    def process(self):
        self._process_split("train")
        self._process_split("val")
        self._process_split("test")

    def _process_split(self, split):
        output_dir_images = os.path.join(self.base_dir, f"{split}/images")
        output_dir_masks = os.path.join(self.base_dir, f"{split}/masks")
        os.makedirs(output_dir_images, exist_ok=True)
        if split != "test":
            os.makedirs(output_dir_masks, exist_ok=True)

        if split != "test":
            annotations_file = os.path.join(self.annotations_dir, "annotations", f"instances_{split}2017.json")
            with open(annotations_file, "r") as f:
                coco_data = json.load(f)

            images_data = coco_data["images"]
            image_id_to_annotations = self._create_annotation_mapping(coco_data["annotations"])

            for image_info in images_data:
                img_filename = image_info["file_name"]
                self.download_image(img_filename, output_dir_images, split)
                create_masks(image_info, image_id_to_annotations, output_dir_masks, self.category_id_to_idx)
        else:
            test_images_url = f"http://images.cocodataset.org/zips/{split}2017.zip"
            test_zip_path = f"{split}2017.zip"
            download_and_extract(test_images_url, test_zip_path, output_dir_images)

    def _create_annotation_mapping(self, annotations):
        image_id_to_annotations = {}
        for annotation in annotations:
            image_id = annotation["image_id"]
            if image_id not in image_id_to_annotations:
                image_id_to_annotations[image_id] = []
            image_id_to_annotations[image_id].append(annotation)
        return image_id_to_annotations

    def save_label_mappings(self):
        annotations_file = os.path.join(self.annotations_dir, "annotations", "instances_train2017.json")
        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        original_label_mapping = {category["name"]: category["id"] for category in coco_data["categories"]}
        self.category_id_to_idx = {}
        label_mapping = {}
        for idx, (name, id) in enumerate(original_label_mapping.items(), start=1):
            self.category_id_to_idx[id] = idx
            label_mapping[name] = idx
        label_mapping["background"] = 0
        self.category_id_to_idx[0] = 0

        save_label_mappings(os.path.join(self.base_dir, "train/masks"), label_mapping, "Train")
        save_label_mappings(os.path.join(self.base_dir, "val/masks"), label_mapping, "Val")
