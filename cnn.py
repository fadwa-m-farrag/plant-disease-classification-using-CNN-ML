import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader 
from torchvision import datasets
from torchvision.transforms import v2 
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np 

device = torch.device('cuda')

train_transform = v2.Compose([
    v2.Resize((128, 128)),
    #v2.Grayscale(num_output_channels=1),
    v2.RandomHorizontalFlip(p=0.2),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
    ])

test_transform = v2.Compose([
     v2.Resize((128,128)),
     v2.ToImage(),
     v2.ToDtype(torch.float32, scale=True)
])

path = '/home/fadwa/Projects/Diploma Project Data/Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/Plant_leave_diseases_dataset_without_augmentation/Apple'
        
full_train_dataset = datasets.ImageFolder(root=path, transform=train_transform) 
full_test_dataset = datasets.ImageFolder(root=path, transform=test_transform)

targets = np.array(full_train_dataset.targets)
indices = np.arange(len(targets))
class_names = full_train_dataset.classes

train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=targets, random_state = 42)

train_dataset = Subset(full_train_dataset, train_idx)
test_dataset = Subset(full_test_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1))
                )
        
    def forward(self, x):
            x = self.conv_layer(x)
            return x.view(x.size(0), -1)

model = CNN().to(device)
model.eval()

def feature_extraction(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            feats = model(images)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.vstack(features), np.array(labels)

x_train, y_train = feature_extraction(train_loader)
x_test, y_test = feature_extraction(test_loader)

pca = PCA(n_components=0.99)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_pca, y_train)

y_pred = knn.predict(x_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, target_names = class_names))

