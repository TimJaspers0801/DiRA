# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""IMPORT PACKAGES"""
import random
from torch.utils.data import Dataset
from PIL import ImageFilter
import torch
from PIL import Image
import zipfile
from pathlib import Path
from typing import Callable
from typing import Tuple
import time


"""DATALOADER FOR .ZIP FILES"""


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ZipDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for loading images from a zip file.

    Args:
        transform (Callable): A callable object (e.g., a torchvision transform) to apply to the loaded images.
        zip_path (Path): The path to the zip file containing the images.
        image_suffix (str): The file suffix (e.g., ".jpg") that valid image files should have.

    Attributes:
        transform (Callable): The provided image transformation function.
        zip_path (Path): The path to the zip file.
        images (list): A list of valid image file names within the zip file.
        image_folder_members (dict): A dictionary mapping image file names to corresponding ZipInfo objects.

    Methods:
        __len__(self): Returns the length of the dataset.
        __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]: Returns an image and a dummy label for a given index.

    Static Methods:
        _valid_member(m: zipfile.ZipInfo, image_suffix: str) -> bool: Checks if a member is a valid image file.
    """
    def __init__(
            self,
            transform: Callable,
            zip_path: Path,
            image_suffix: str,
    ):

        # Assign variables
        self.transform = transform
        self.zip_path = zip_path
        self.images = []

        # Load the zip file
        image_zip = zipfile.ZipFile(self.zip_path)

        # Get the members of the zip file
        self.image_folder_members = {
            str(Path(m.filename)): m
            for m in sorted(image_zip.infolist(), key=lambda x: x.filename)
        }

        # Get the image names from the zip file, check whether they are valid
        for image_name, m in self.image_folder_members.items():
            if not self._valid_member(
                    m, image_suffix
            ):
                continue
            self.images.append(image_name)

    @staticmethod
    def _valid_member(
            m: zipfile.ZipInfo,
            image_suffixes: list,
    ):
        """Returns True if the member is valid based on the list of suffixes"""
        return (
                any(m.filename.endswith(suffix) for suffix in image_suffixes)
                and not m.is_dir()
        )
    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the items for a dataloader object"""

        # Open the zip file
        with zipfile.ZipFile(self.zip_path) as image_zip:

            # Open the image data from the zip file based on index
            with image_zip.open(
                    self.image_folder_members[self.images[index]].filename
            ) as image_file:
                image = Image.open(image_file).convert("RGB")

        # Create dummy label to return DINO dataloader
        label = torch.tensor(0)

        # Apply torchvision transforms if defined
        if self.transform:
            image = self.transform(image)

        return image, label


"""FUNCTION FOR CONCATENATING .ZIP DATASETS"""

def concat_zip_datasets(
        parent_folder: str,
        transform: Callable,
        image_suffix: list = ['.png', 'jpg'],
        datasets: list = None,
):
    """
        Concatenates multiple ZipDatasets into a single ConcatDataset.

        Args:
            parent_folder (str): The path to the parent folder containing multiple zip files to be combined.
            transform (Callable): A callable object (e.g., a torchvision transform) to apply to the loaded images.
            image_suffix (str, optional): The file suffix (e.g., ".jpg") that valid image files should have.
                Defaults to '.png'.

        Returns:
            torch.utils.data.ConcatDataset: A ConcatDataset containing all the ZipDatasets.

        Note:
            To use this function, provide the path to the parent folder containing the zip files you want to combine.
            You can also specify a custom image_suffix and transformation function.
        """

    # Create list of zip folders
    zip_folders = list(Path(parent_folder).iterdir())

    if datasets is not None:
        included_folders = []
        for dataset in datasets:
            for zip_folder in zip_folders:
                if dataset in zip_folder.name:
                    included_folders.append(zip_folder)
    else:
        included_folders = zip_folders

    # Construct datasets for each zip folder
    dataset = [
        ZipDataset(
            transform=transform,
            zip_path=zip_folder,
            image_suffix=image_suffix)
        for zip_folder in included_folders
    ]

    # Concatenate the datasets
    dataset = torch.utils.data.ConcatDataset(dataset)

    return dataset


"""FUNCTION FOR CONCATENATING .ZIP DATASETS"""

def custom_collate_fn(batch):
    # Transpose the list of samples to have the batch dimension within the list
    batch = list(map(list, zip(*batch)))
    return [torch.stack(item) for item in batch]
