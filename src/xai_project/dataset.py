"""
dataset.py - DomainNet dataset class and image transforms.

Provides DomainNetDataset (reads list of (path, label) pairs) and
get_transforms() for train/val/test augmentation pipelines.
"""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split):
    """
    Return image transforms for the given split.

    Args:
        split: one of 'train', 'val', 'test'

    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class DomainNetDataset(Dataset):
    """
    Dataset for DomainNet images.

    Args:
        samples: list of (image_path, label) tuples/lists
        transform: torchvision transform to apply to each image
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
