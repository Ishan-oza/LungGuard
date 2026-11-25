import os
from collections import defaultdict

# Dynamic dataset path (fallback to local data/ or environment variable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.environ.get("CHX_DATASET_DIR", os.path.join(BASE_DIR, "data"))

counts = defaultdict(lambda: {"train": 0, "test": 0})
image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

print(f"🔍 Inspecting dataset at: {dataset_path}")

for split in ["train", "test"]:
    split_path = os.path.join(dataset_path, split)
    if not os.path.exists(split_path):
        continue
    for root, dirs, files in os.walk(split_path):
        parts = os.path.normpath(root).split(os.sep)
        if len(parts) > 0 and parts[-1].lower() not in ["train", "test"]:
            category = parts[-1]
            num_imgs = len([f for f in files if f.lower().endswith(image_exts)])
            counts[category][split] += num_imgs

print("\n📊 DATASET STATISTICS")
print("───────────────────────────────")
total_images = 0
for category, split_counts in counts.items():
    total_cat = split_counts["train"] + split_counts["test"]
    total_images += total_cat
    print(f"\nClass: {category}")
    print(f"  ➤ Train: {split_counts['train']} images")
    print(f"  ➤ Test : {split_counts['test']} images")
    print(f"  ➤ Total: {total_cat} images")

print("\n───────────────────────────────")
print(f"Total images across all classes: {total_images}\n")

if total_images > 0:
    print("📈 Percentage Distribution")
    for category, split_counts in counts.items():
        total_cat = split_counts["train"] + split_counts["test"]
        percent = (total_cat / total_images) * 100
        print(f"  {category}: {percent:.2f}%")
