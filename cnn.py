import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, Subset, DataLoader 
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split 
import numpy as np 

device = torch.device('cuda')

# splitting the folders into train & test datasets without messing up
# the folder structure of my dataset 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor
    ])

dataset = datasets.ImageFolder(root='/home/fadwa/Projects/Diploma Project Data/Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network', transform=transform) 

targets = np.array(dataset.targets)
indices = np.arange(len(targets))

train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=targets, random_state = 42)

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


