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
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_data = pd.read_csv(os.path.join("..", "train_data.csv"))
validation_data = pd.read_csv(os.path.join("..", "validation_data.csv"))
test_data = pd.read_csv(os.path.join("..", "test_data.csv"))
dir_path = "./Fitzpatric_subset/"

skin_tones = ["Light skin. ", "Dark skin. "]


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

        description = self.df.iloc[idx]["Most Present Concepts"]
        if self.transform:
            image = self.transform(image)
        return image, int(skin), int(lesion), description


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

batch_size = 16
num_workers = 2

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
checkpoint_path = f"MONET_MostPresent_desc_vit32b1_skin_{optimizer_type}_{lr}_{scheduler_type}_best.pth"
output_filename = (
    f"MONET_MostPresent_desc_vit32b1_skin_{optimizer_type}_{lr}_{scheduler_type}.txt"
)


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, _, labels, descriptions in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, descriptions)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_clip)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (
            images,
            _,
            labels,
            descriptions,
        ) in validation_loader:  # Use validation dataloader
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, descriptions)
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

print("Training Complete! Test starts")

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
    for images, skins, labels, descriptions in test_loader:
        images = images.to(device)
        outputs = model(images, descriptions)
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

with open(output_filename, "w") as file:
    file.write("Test output:\n")
    file.write(f"\nvalidation loss = {val_losses}")
    file.write(f"\nvit = 32B")
    file.write(f"\ntokenizer = {text_model}")
    file.write(f"\nintegrate_way = concatenate")
    file.write(f"\noptimizer = {optimizer_type}")
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
