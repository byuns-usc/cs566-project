import os
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, base_dir):
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
