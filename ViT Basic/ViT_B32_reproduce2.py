#! /usr/bin/env python3.12

from torchvision.models import vit_b_32, ViT_B_32_Weights
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import pandas as pd
import os
from PIL import Image


model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


grad_norm_clip = 1

lr = 0.001
# scheduler_type = "CosineAnnealingLR"
scheduler_type = "fixed"
optimizer_type = "SGD_Momentum"
# optimizer_type = "Adam"

checkpoint = f"torchvision_vit32b_skin_{optimizer_type}_{lr}_{scheduler_type}"
checkpoint_path = checkpoint + "_best.pth"
output_file = checkpoint + ".txt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))


test_data = pd.read_csv("../test_data.csv")
dir_path = "../Fitzpatric_subset/"


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_filename = self.df.iloc[idx]["image_path"] + ".jpg"
        skin = self.df.iloc[idx]["skin_color"]
        lesion = self.df.iloc[idx]["lesion"]
        img_path = os.path.join(dir_path, img_filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, skin, lesion


mean = [0.6373, 0.4955, 0.4401]
std = [0.2125, 0.1923, 0.1954]
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

test_dataset = SkinDataset(test_data, transform)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

skin_metrics2 = {
    i: {"total": 0, "correct": 0, "accuracy": None, "predict": list(), "true": list()}
    for i in range(1, 3)
}
skin_metrics6 = {
    i: {"total": 0, "correct": 0, "accuracy": None, "predict": list(), "true": list()}
    for i in range(1, 7)
}
model.eval()
with torch.no_grad():
    for images, skins, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.argmax(dim=-1).squeeze().tolist()
        skins = skins.tolist()
        labels = labels.tolist()
        for output, skin, label in zip(outputs, skins, labels):
            skin_metrics6[skin]["total"] += 1
            skin_metrics6[skin]["correct"] += output == label
            skin_metrics6[skin]["predict"].append(output)
            skin_metrics6[skin]["true"].append(label)
            skin_metrics2[1 if skin <= 4 else 2]["total"] += 1
            skin_metrics2[1 if skin <= 4 else 2]["correct"] += output == label
            skin_metrics2[1 if skin <= 4 else 2]["predict"].append(output)
            skin_metrics2[1 if skin <= 4 else 2]["true"].append(label)
        for key, item in skin_metrics2.items():
            item["accuracy"] = item["correct"] / item["total"]
        for key, item in skin_metrics6.items():
            item["accuracy"] = item["correct"] / item["total"]

print("2 skin results:")
for key, item in skin_metrics2.items():
    print(
        f"skin{key} total={item['total']}, correct={item['correct']}, accuracy={item['accuracy']}"
    )

print("6 skin results:")
for key, item in skin_metrics6.items():
    print(
        f"skin{key} total={item['total']}, correct={item['correct']}, accuracy={item['accuracy']}"
    )

with open(output_file, "w") as file:
    file.write("Test output:\n")
    # file.write(f"\nvalidation loss = {val_losses}")
    file.write(f"\nlearning_rate = {0.001}")
    file.write(f"\nweight_decay = {1e-4}")
    file.write(f"\nscheduler = {scheduler_type}")
    # file.write(f"\nbatch_size = {batch_size}")
    # file.write(f"\nepochs = {epoch}")
    file.write(f"\nmax_stop_count = {5}")
    file.write(f"\ngrad_norm_clip = {grad_norm_clip}")
    for skin, metrics in skin_metrics6.items():
        file.write(
            f"\nskin tone {skin} true label:{metrics['true']}\nskin tone {skin} predicted label:{metrics['predict']}",
        )

    for skin, metrics in skin_metrics2.items():
        file.write(
            f"\nbinary skin tone {skin} true label:{metrics['true']}\nbinary skin tone {skin} predicted label:{metrics['predict']}",
        )
