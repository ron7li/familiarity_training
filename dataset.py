
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform, noise=False, noise_ratio=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.noise = noise
        self.noise_ratio = noise_ratio
        
        # If noise is enabled, pre-generate noisy versions of all images
        if self.noise:
            self.noisy_images = []
            for image_path in self.image_paths:
                image = Image.open(image_path).convert("RGB")
                noisy_image = self.add_salt_and_pepper_noise(image)
                self.noisy_images.append(noisy_image)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.noise:
            # Use pre-generated noisy images
            image_2 = self.noisy_images[idx]
        else:
            image_2 = image.copy()
            
        label = self.labels[idx]

        return self.transform(image), self.transform(image_2), label

    def add_salt_and_pepper_noise(self, image):
        """
        Add salt and pepper noise to the image
        """
        image_array = np.array(image, dtype=np.uint8)
        h, w, c = image_array.shape  # Get image dimensions
        total_pixels = h * w
        num_noise_pixels = int(self.noise_ratio * total_pixels)  # Calculate the number of pixels to change

        # Generate all pixel positions
        all_positions = np.arange(total_pixels)
        
        # Randomly select noise pixel positions, ensuring no duplicates
        noise_positions = np.random.choice(all_positions, size=num_noise_pixels, replace=False)
        
        # Convert one-dimensional indices to two-dimensional coordinates
        noise_coords = np.unravel_index(noise_positions, (h, w))
        
        # Add salt noise (white) to the first half, and pepper noise (black) to the second half
        salt_count = num_noise_pixels // 2
        pepper_count = num_noise_pixels - salt_count
        
        # Add salt noise (white)
        for i in range(salt_count):
            x, y = noise_coords[0][i], noise_coords[1][i]
            image_array[x, y] = [255, 255, 255]
        
        # Add pepper noise (black)
        for i in range(salt_count, num_noise_pixels):
            x, y = noise_coords[0][i], noise_coords[1][i]
            image_array[x, y] = [0, 0, 0]

        return Image.fromarray(image_array)
    