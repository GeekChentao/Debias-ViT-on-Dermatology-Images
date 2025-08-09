#! /usr/bin/env python3.12

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vit_b_32, ViT_B_32_Weights
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel
import psutil
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_data = pd.read_csv(os.path.join("..", "train_data.csv"))
validation_data = pd.read_csv(os.path.join("..", "validation_data.csv"))
test_data = pd.read_csv(os.path.join("..", "test_data.csv"))
dir_path = "../Fitzpatric_subset/"
skin_tones = ["Light skin. ", "Dark skin. "]
track_memory = False


class MemoryTracker:
    """Track memory usage throughout training and testing phases"""

    def __init__(self):
        self.max_gpu_memory = 0
        self.max_cpu_memory = 0
        self.total_params = 0

    def log_memory(self, phase="", epoch=None, batch=None):
        """Log current memory usage"""
        self.total_params = max(
            self.total_params, sum(p.numel() for p in model.parameters())
        )

        # Get current memory
        gpu_memory = (
            torch.cuda.memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0
        )
        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        # Update max memory
        self.max_gpu_memory = max(self.max_gpu_memory, gpu_memory)
        self.max_cpu_memory = max(self.max_cpu_memory, cpu_memory)

        return gpu_memory, cpu_memory

    def print_summary(self):
        """Print memory usage summary"""
        print(f"\n=== Memory Usage Summary ===")
        print(f"Total Parameters: {self.total_params:,}")
        print(f"Max GPU Memory: {self.max_gpu_memory:.2f} MB")
        print(f"Max CPU Memory: {self.max_cpu_memory:.2f} MB")
        print(
            f"Parameter Memory (estimated): {self.total_params * 4 / (1024 * 1024):.2f} MB"
        )


if track_memory:
    memory_tracker = MemoryTracker()
    print("Memory tracking is enabled")
else:
    memory_tracker = None
    print("Time tracking is enabled")


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
        description = skin_tones[0 if int(skin) <= 4 else 1]
        if self.transform:
            image = self.transform(image)
        return image, skin, lesion, description


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_model = "sentence-transformers/all-MiniLM-L6-v2"


# Vit with Transformer Model Constructure
class VitTransformerClassifier(nn.Module):
    def __init__(self):
        super(VitTransformerClassifier, self).__init__()

        self.text_model = AutoModel.from_pretrained(text_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_feature_dim = 384

        self.vit_model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT).to(device)
        self.vit_feature_dim = self.vit_model.heads.head.in_features
        self.vit_model.heads.head = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.vit_feature_dim + self.text_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, image, text):
        text_tokenized = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        text_tokenized = {k: v.to(device) for k, v in text_tokenized.items()}
        text_features = self.text_model(**text_tokenized).last_hidden_state[
            :, 0, :
        ]  # CLS token embedding

        img_features = self.vit_model(image)
        combined_features = torch.cat((img_features, text_features), dim=1)
        output = self.fc(combined_features)
        return output


model = VitTransformerClassifier().to(device)
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
scheduler = scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler_type = "CosineAnnealingLR"

# scheduler = None
if not scheduler:
    scheduler_type = "fixed"

grad_norm_clip = 1
checkpoint_path = (
    f"SkinDesc_vit32b1_skin_{optimizer_type}_{lr}_{scheduler_type}_best.pth"
)
output_filename = f"SkinDesc_vit32b1_skin_{optimizer_type}_{lr}_{scheduler_type}.txt"

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (images, _, labels, descriptions) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, descriptions)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_clip)
        optimizer.step()
        train_loss += loss.item()

        if track_memory:
            if batch_idx % 8 == 0:
                memory_tracker.log_memory(
                    phase="training_batch", epoch=epoch, batch=batch_idx
                )

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, _, labels, descriptions) in enumerate(
            validation_loader
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, descriptions)
            loss = criterion(outputs, labels)
            # print(loss)
            val_loss += loss.item()

            if track_memory:
                if batch_idx % 8 == 0:
                    memory_tracker.log_memory(
                        phase="validation_batch", epoch=epoch, batch=batch_idx
                    )

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

    if track_memory:
        break

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

start_time = time.perf_counter()
with torch.no_grad():
    for batch_idx, (images, skins, labels, descriptions) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images, descriptions)
        outputs = outputs.argmax(dim=-1).squeeze().tolist()

        if track_memory:
            if batch_idx % 8 == 0:
                memory_tracker.log_memory(phase="testing_batch", batch=batch_idx)

end_time = time.perf_counter()
print(f"Testing time: {(end_time - start_time)/len(test_dataset)*1000:.2f} ms/sample")

if track_memory:
    memory_tracker.print_summary()
