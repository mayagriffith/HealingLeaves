from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class DamagedHealthyDataset(Dataset):
    def __init__(self, root_damaged, root_healthy, transform=None):
        self.root_damaged = root_damaged # Root directory for damaged leaf images
        self.root_healthy = root_healthy  # Root directory for healthy images
        self.transform = transform  # Transformation function to apply to images

        self.damaged_images = [file for file in os.listdir(root_damaged) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.healthy_images = [file for file in os.listdir(root_healthy) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # The length of the dataset is determined by the maximum number of images in either category
        self.length_dataset = max(len(self.damaged_images), len(self.healthy_images)) # 1000, 1500
        self.damaged_len = len(self.damaged_images) # 1000
        self.healthy_len = len(self.healthy_images) # 1500

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        # Get the filenames of damagedd and healthy images based on the current index
        damaged_leaf_img = self.damaged_images[index % self.damaged_len]
        healthy_leaf_img = self.healthy_images[index % self.healthy_len]
        
        # Construct the full file paths for damaged and healthy leaf images
        damaged_path = os.path.join(self.root_damaged, damaged_leaf_img)
        healthy_path = os.path.join(self.root_healthy, healthy_leaf_img)

        # Load and convert damaged and healthy images to NumPy arrays
        damaged_leaf_img = np.array(Image.open(damaged_path).convert("RGB"))
        healthy_leaf_img = np.array(Image.open(healthy_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=damaged_leaf_img, image0=healthy_leaf_img)
            damaged_leaf_img = augmentations["image"]
            healthy_leaf_img = augmentations["image0"]

        return damaged_leaf_img, healthy_leaf_img

