from PIL import Image
import numpy as np
import torch
import os
from configs import *
import cv2
from torch.utils.data import Dataset
from torchvision import transforms 

data_transforms = {
        "images": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "masks": transforms.Compose([
            transforms.ToTensor()
        ])
    }


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class MedicalDataset(Dataset):
    def __init__(self, root):
        super(MedicalDataset, self).__init__()
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, image_dir))))
        self.masks = list(sorted(os.listdir(os.path.join(root, mask_dir))))
        self.transformer = data_transforms

    def __getitem__(self, idx):
        # load images ad masks
        name_img = self.imgs[idx]
        #print("name_img", name_img)
        img_path = os.path.join(self.root, image_dir, name_img)

        mask_path = os.path.join(self.root, mask_dir, name_img)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(mask*multiply_value)
        
        img = self.transformer["images"](img)
        mask = self.transformer["masks"](mask)
        return img, mask, name_img

    def __len__(self):
        return len(self.imgs)

class MedicalDataset_Test(Dataset):
    def __init__(self, root):
        super(MedicalDataset_Test, self).__init__()
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, image_dir))))
        self.masks = list(sorted(os.listdir(os.path.join(root, mask_dir))))
        self.transformer = data_transforms

    def __getitem__(self, idx):
        # load images ad masks
        name_img = self.imgs[idx]
        img_path = os.path.join(self.root, image_dir, name_img)

        mask_path = os.path.join(self.root, mask_dir, name_img)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        img = cv2.resize(img, (512, 512))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(mask*multiply_value)
        
        img = self.transformer["images"](img)
        mask = self.transformer["masks"](mask)
        return img, mask, name_img

    def __len__(self):
        return len(self.imgs)

