import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class HC18Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))  # Sort to align images and masks
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load images and masks
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Apply transformations
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

# Define transformation of images and mask
image_transformation = transforms.Compose([
    transforms.Grayscale(num_output_channels= 1), # Ensure single channel
    transforms.Resize((256, 256)),
    transforms.ToTensor() # Convert it to tensor
])

mask_transformation = transforms.Compose([
    transforms.Grayscale(num_output_channels= 1),
    transforms.Resize((256, 256)),
    transforms.ToTensor() # Convert it to tensor
])

def get_dataloader(image_dir, mask_dir, batch_size=8):
    dataset = HC18Dataset(image_dir, mask_dir, image_transformation, mask_transformation)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
