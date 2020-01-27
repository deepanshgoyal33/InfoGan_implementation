import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
import torch.transforms as transforms
import torch.Datasets as Datasets

#celeb a  dataset
root_dir ='./Dataset/'
def dataloading(batch_size):

    transform = transforms.Compose({
        transforms.resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),
            (0.5,0.5,0.5))
    })
    dataset = datasets.ImageFolder(root=root_dir+'img_align_celeba/',transform = transform)

    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch)

    return dataloader