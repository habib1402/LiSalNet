import torch
import cv2
import torch.utils.data
import torch.nn.functional as F
import random
from pathlib import Path


def resolve_manifest_path(file_root, entry):
    """Resolve both ordinary and legacy leading-slash manifest entries.

    Some of the manifests used for the paper store root-relative paths such as
    ``/NJU2K/RGB/0001.jpg``.  ``pathlib`` normally treats those as filesystem
    absolute and discards ``file_root``.  A genuinely absolute path is still
    honoured when it exists; otherwise the leading slash is interpreted as
    relative to the supplied data root.
    """
    root = Path(file_root).expanduser()
    path = Path(entry).expanduser()
    if path.is_absolute() and path.exists():
        return path
    return root / str(entry).lstrip('/\\')


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, dataset, file_root='data/', transform=None, use_depth=False, depth_postfix="_depth.png"):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        with open(Path(file_root) / (dataset + '.txt')) as handle:
            self.file_list = [line.strip() for line in handle if line.strip()]
        if not self.file_list:
            raise ValueError('Dataset manifest is empty: ' + dataset)
        if any(len(line.split()) != 2 for line in self.file_list):
            raise ValueError('Each manifest line must contain: image_path mask_path')
        self.images = [str(resolve_manifest_path(file_root, x.split()[0])) for x in self.file_list]
        self.gts = [str(resolve_manifest_path(file_root, x.split()[1])) for x in self.file_list]
        self.transform = transform
        self.use_depth = use_depth
        self.depth_postfix = depth_postfix

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label_name = self.gts[idx]
        depth_name = str(Path(image_name).with_suffix('')) + self.depth_postfix
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        depth = cv2.imread(depth_name, 0) if self.use_depth else None
        for name, value in [(image_name, image), (label_name, label)] + ([(depth_name, depth)] if self.use_depth else []):
            if value is None:
                raise FileNotFoundError('Could not read image: ' + name)
        if image.shape[:2] != label.shape[:2] or (depth is not None and depth.shape != label.shape):
            raise ValueError('RGB, mask and depth must have matching source dimensions: ' + image_name)
        if self.transform:
            [image, label, depth] = self.transform(image, label, depth)
        if self.use_depth:
            return image, label, depth
        else:
            return image, label

    def get_img_info(self, idx):
        img = cv2.imread(self.images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
