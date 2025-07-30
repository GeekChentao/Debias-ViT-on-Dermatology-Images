#! /usr/bin/env python3.12

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vit_l_32, ViT_L_32_Weights
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time

train_data = pd.read_csv("../train_data.csv")
validation_data = pd.read_csv("../validation_data.csv")
test_data = pd.read_csv("../test_data.csv")
dir_path = "../Fitzpatric_subset/"


class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_filename = self.df.iloc[idx]["image_path"]
        img_filename += ".jpg"
        img_path = os.path.join("..", dir_path, img_filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image


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
        img_path = os.path.join("..", dir_path, img_filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, skin, lesion


def calculate_mean_std(loader):
    print("Calculatating mean and std for trainning images")
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for images in tqdm(loader):
        batch_samples = images.shape[0]  # Get batch size
        images = images.view(batch_samples, 3, -1)  # Reshape to (B, C, H*W)
        mean += images.mean(dim=[0, 2]) * batch_samples  # Sum of mean per channel
        std += images.std(dim=[0, 2]) * batch_samples  # Sum of std per channel
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.tolist(), std.tolist()


image_dataset = ImageDataset(train_data, None)
dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)
mean, std = calculate_mean_std(dataloader)
print(f"mean = {[round(m, 4) for m in mean]}; std = {[round(s, 4) for s in std]}")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # ViT expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

batch_size = 32
num_workers = 4

train_dataset = SkinDataset(train_data, transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
validation_dataset = SkinDataset(validation_data, transform)
validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_dataset = SkinDataset(test_data, transform)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, 2)  # Assuming 2 classes
criterion = nn.CrossEntropyLoss()

patience = 5
num_epochs = 100
best_val_loss = float("inf")
counter = 0
val_losses = list()

# Adam Optimizer
# lr = 0.001
# weight_dacay = 1e-4
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_dacay)
# optimizer_type = "Adam"

# AdamW Optimizer
# lr = 0.001
# weight_dacay = 1e-4
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_dacay)
# optimizer_type = "AdamW"

# SGD with Momentum Optimizer
lr = 0.001
weight_dacay = 1e-4
momentum = 0.9
optimizer = optim.SGD(
    model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_dacay
)
optimizer_type = "SGD_Momentum"

# Add LR Scheduler
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# scheduler_type = "StepLR"

# Add Cosine Annealing LR Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler_type = "CosineAnnealingLR"

# scheduler = None
# if not scheduler:
#     scheduler_type = "fixed"

grad_norm_clip = 1
checkpoint_path = (
    f"torchvision_vit32l_skin_{optimizer_type}_{lr}_{scheduler_type}_best.pth"
)
output_file = f"torchvision_vit32b_skin_{optimizer_type}_{lr}_{scheduler_type}.txt"

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, _, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_clip)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _, labels in validation_loader:  # Use your validation dataloader
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print(loss)
            val_loss += loss.item()

    val_loss /= len(validation_loader)
    val_losses.append(val_loss)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )
    if scheduler:
        scheduler.step()

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0  # Reset counter if validation loss improves
        torch.save(model.state_dict(), checkpoint_path)  # Save the best model
        print("New model path saved")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break  # Stop training

end_time = time.time()
print(
    f"Training time: {end_time - start_time:.2f} seconds\nTraining Complete! Test starts"
)

skin_metrics2 = {
    i: {"total": 0, "correct": 0, "accuracy": None, "predict": list(), "true": list()}
    for i in range(1, 3)
}
skin_metrics6 = {
    i: {"total": 0, "correct": 0, "accuracy": None, "predict": list(), "true": list()}
    for i in range(1, 7)
}
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    file.write(f"\nvalidation loss = {val_losses}")
    file.write(f"\nlearning_rate = {lr}")
    file.write(f"\nweight_decay = {weight_dacay}")
    file.write(f"\nscheduler = {scheduler_type}")
    file.write(f"\nbatch_size = {batch_size}")
    file.write(f"\nepochs = {epoch}")
    file.write(f"\nmax_stop_count = {patience}")
    file.write(f"\ngrad_norm_clip = {grad_norm_clip}")
    for skin, metrics in skin_metrics6.items():
        file.write(
            f"\nskin tone {skin} true label:{metrics['true']}\nskin tone {skin} predicted label:{metrics['predict']}",
        )

    for skin, metrics in skin_metrics2.items():
        file.write(
            f"\nbinary skin tone {skin} true label:{metrics['true']}\nbinary skin tone {skin} predicted label:{metrics['predict']}",
        )
