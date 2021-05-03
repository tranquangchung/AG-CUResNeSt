# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import evaluation_in_medical_cunet
import utils
import pdb
from datasets import MedicalDataset, MedicalDataset_Test
from configs import *
from  models.unet import CUnet_Resnest
from collections import OrderedDict
from loss_factory import Loss_Factory
import glob


def main():
    torch.manual_seed(0)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define dataset and dataloader
    dataset_test = MedicalDataset_Test(test_path)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1)

    # define the model
    model = CUnet_Resnest(encoder, encoder_weights='imagenet',couple_unet=True)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(path_checkpoint))
    evaluation_in_medical_cunet(model, data_loader_test, device=device, couple_unet=couple_unet)
    
if __name__ == "__main__":
    main()

