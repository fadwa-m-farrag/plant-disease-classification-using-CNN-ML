import torch
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler 
from torchvision import datasets
from torchvision.transforms import v2 
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np 
from collections import Counter
import joblib 
import mlflow 
import mlflow.sklearn
from mlflow.models import infer_signature
import os
from cnn import CNN 

mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_ENABLE_ARTIFACTS_STORAGE"] = "false"
device = torch.device('cuda')

train_transform = v2.Compose([
    #v2.Resize((128, 128)),
    #v2.Grayscale(num_output_channels=1),
    #v2.RandomHorizontalFlip(p=0.2),
    #v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
    ])

test_transform = v2.Compose([
     #v2.Resize((128,128)),
     v2.ToImage(),
     v2.ToDtype(torch.float32, scale=True)
])

path = '/home/fadwa/Projects/Diploma Project Data/Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/Plant_leave_diseases_dataset_without_augmentation/Apple'
        
full_train_dataset = datasets.ImageFolder(root=path, transform=train_transform) 
full_test_dataset = datasets.ImageFolder(root=path, transform=test_transform)

targets = np.array(full_train_dataset.targets)
indices = np.arange(len(targets))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, stratify=targets, random_state=42
)
train_targets = targets[train_idx]
class_counts = Counter(train_targets)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = np.array([class_weights[t] for t in train_targets])
class_names = full_train_dataset.classes
num_classes = len(class_counts)

train_dataset = Subset(full_train_dataset, train_idx)
test_dataset = Subset(full_test_dataset, test_idx)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  
    replacement=True  
)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

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
with mlflow.start_run():
    xgb = XGBClassifier(objective='multi:softmax', num_class=num_classes)
    xgb.fit(x_train_pca, y_train)

    y_pred = xgb.predict(x_test_pca)

    mlflow.log_param("model_type", "XGB Classifier")
    mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred))
    signature = infer_signature(x_test, y_pred)
    #input_example = x_test[:1]
    mlflow.sklearn.log_model(sk_model=xgb, name="XGBClassifier", signature=signature)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    report = classification_report(y_test,y_pred, target_names = class_names)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

torch.save(model.state_dict(), 'cnn_feature_extractor.pth')

joblib.dump(pca, 'pca_model.pkl')
joblib.dump(xgb, 'ml_classifier.pkl')

joblib.dump(class_names, 'class_names.pkl')
