import os
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

def count_files_by_prefix(root_path):
    prefix_counts = defaultdict(int)

    for dirpath, _, filenames in os.walk(root_path):
        folder_name = os.path.basename(dirpath)
        if "_" in folder_name:
            prefix = folder_name.split("_")[0]  # e.g., 'Apple' from 'Apple_example1'
            prefix_counts[prefix] += sum(
                os.path.isfile(os.path.join(dirpath, f)) for f in filenames
            )
    
    return prefix_counts

def plot_prefix_counts(prefix_counts):
    prefixes = list(prefix_counts.keys())
    counts = [prefix_counts[p] for p in prefixes]

    plt.figure(figsize=(10, 6))
    plt.bar(prefixes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('File Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

dataset = '/home/fadwa/Projects/Diploma Project Data/Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/Plant_leave_diseases_dataset_without_augmentation'
prefix_counts = count_files_by_prefix(dataset)
plot_prefix_counts(prefix_counts)

def calc_dimensions(path):
    dimensions = []
    for plant in os.listdir(path):
        plant_path = os.path.join(path, plant)
        if not os.path.isdir(plant_path):
            continue

        has_subfolders = any(
            os.path.isdir(os.path.join(plant_path, f)) for f in os.listdir(plant_path)
        )

        if has_subfolders:
            for condition in os.listdir(plant_path):
                condition_path = os.path.join(plant_path, condition)
                if not os.path.isdir(condition_path):
                    continue

                for fname in os.listdir(condition_path):
                    if fname.lower().endswith('.jpg'):
                        fpath = os.path.join(condition_path, fname)
                        img = Image.open(fpath)
                        dimensions.append(img.size)

        else:
            for fname in os.listdir(plant_path):
                if fname.lower().endswith('.jpg'):
                    fpath = os.path.join(plant_path, fname)
                    img = Image.open(fpath)
                    dimensions.append(img.size)
    return dimensions

dims = calc_dimensions(dataset)
print(f"Total images processed: {len(dims)}")

widths, heights = zip(*dims)
plt.hist(widths, bins=20, alpha=0.6, label='Widths')
plt.hist(heights, bins=20, alpha=0.6, label='Heights')
plt.legend()
plt.title("Image Dimension Distribution")
plt.xlabel("Pixels")
plt.ylabel("Number of Images")
plt.show()
