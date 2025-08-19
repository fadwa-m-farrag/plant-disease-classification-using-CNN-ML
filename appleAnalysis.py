import os
import matplotlib.pyplot as plt 
from pathlib import Path 

dataset = '/home/fadwa/Projects/Diploma Project Data/Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/Plant_leave_diseases_dataset_without_augmentation/Apple'

image_counts = {}

for subdir in os.listdir(dataset):
    subdir_path = os.path.join(dataset, subdir)
    if os.path.isdir(subdir_path):
        count = len([
            f for f in os.listdir(subdir_path)
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
            ])
    image_counts[subdir] = count

plt.figure(figsize=(10,6))
plt.bar(image_counts.keys(), image_counts.values())
plt.show()
