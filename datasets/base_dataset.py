import os
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, base_dir: str):
        """
        Initialize the dataset with a base directory.

        Args:
            base_dir (str): The base directory where the dataset will be stored.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    @abstractmethod
    def download(self):
        """
        Download the dataset if not already present.
        """
        pass

    @abstractmethod
    def process(self):
        """
        Process the downloaded data into a format suitable for training.
        """
        pass

    @abstractmethod
    def save_label_mappings(self):
        """
        Save the label mappings for the dataset.
        """
        pass
