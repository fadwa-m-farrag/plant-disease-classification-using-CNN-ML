import torch
import numpy as np 
from sklearn.metrics import accuracy_score, classification_report
import joblib
from cnn import CNN
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device('cuda')
model = CNN().to(device)
model.load_state_dict(torch.load("model files/cnn.pth", map_location=device))
model.eval()

pca = joblib.load("model files/pca.pkl")
clf = joblib.load("model files/xgb.pkl")

test_transform = v2.Compose([
     v2.ToImage(),
     v2.ToDtype(torch.float32, scale=True)
])

path = '/home/fadwa/Projects/Diploma Project Data/Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/Plant_leave_diseases_dataset_without_augmentation/Apple'

dataset = datasets.ImageFolder(root=path, transform=test_transform)

targets = np.array(dataset.targets)
indices = np.arange(len(targets))
_, test_idx = train_test_split(
    indices, test_size=0.2, stratify=targets, random_state=42
)

test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            feats = model(images).cpu().numpy()
            features.append(feats)
            labels.extend(lbls.numpy())
    return np.vstack(features), np.array(labels)

features, labels = extract_features(model, test_loader)
features = pca.transform(features)

predictions = clf.predict(features)

print("Accuracy:", accuracy_score(labels, predictions))
print(classification_report(labels, predictions))
