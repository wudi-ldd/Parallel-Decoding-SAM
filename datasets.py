import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# Read split files (train/val file lists)
def read_split_files(file_path):
    with open(file_path, 'r') as f:
        file_names = f.read().strip().split('\n')
    return file_names

# Dataset class for segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, sam_model, file_list, mask_size=(1024, 1024), device='cpu'):
        self.image_dir = image_dir  # Directory containing images
        self.mask_dir = mask_dir    # Directory containing masks
        self.sam_model = sam_model  # SAM model instance
        self.mask_size = mask_size  # Size to resize masks
        self.device = device        # Device for tensor operations
        # Filter image files based on file_list
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and f.replace('.png', '') in file_list]

    def __len__(self):
        # Return the total number of samples
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image file name
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Load and preprocess mask
        mask_file = image_file
        mask_path = os.path.join(self.mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.mask_size, interpolation=cv2.INTER_NEAREST)

        # Convert image to tensor and preprocess with SAM
        input_image_torch = torch.as_tensor(image, dtype=torch.float32).to(self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()  # [C, H, W]
        input_image = self.sam_model.preprocess(input_image_torch.to(self.device))

        # Convert mask to tensor
        mask = torch.as_tensor(mask, dtype=torch.float32).to(self.device)  # Single-channel float

        return input_image, mask