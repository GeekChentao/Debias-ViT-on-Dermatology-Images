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
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_data = pd.read_csv(os.path.join("..", "train_data.csv"))
validation_data = pd.read_csv(os.path.join("..", "validation_data.csv"))
test_data = pd.read_csv(os.path.join("..", "test_data.csv"))
dir_path = "../Fitzpatric_subset/"

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

        description = (
            skin_tones[0 if int(skin) <= 4 else 1]
            + self.df.iloc[idx]["Gemini Description"]
        )
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

batch_size = 8
num_workers = 1

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

# text_model = "sentence-transformers/all-MiniLM-L6-v2"
text_model = "bert-base-uncased"


# Vit with Transformer Model Constructure
class VitTransformerClassifier(nn.Module):
    def __init__(self):
        super(VitTransformerClassifier, self).__init__()

        self.text_model = AutoModel.from_pretrained(text_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_feature_dim = 768

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


checkpoint = "GeminiDesc_vit32b2_skin_SGD_Momentum_0.001_CosineAnnealingLR"
checkpoint_path = f"{checkpoint}_best.pth"
output_file = f"{checkpoint}.txt"

model = VitTransformerClassifier().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

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

with open(output_file, "w") as file:
    file.write("Test output:\n")
    # file.write(f"\nvalidation loss = {val_losses}")
    file.write(f"\nvit = 32B")
    file.write(f"\ntokenizer = {text_model}")
    file.write(f"\nintegrate_way = concatenate")
    file.write(f"\noptimizer = SGD_Momentum")
    file.write(f"\nlearning_rate = {0.001}")
    file.write(f"\nweight_decay = {1e-4}")
    file.write(f"\nscheduler = CosineAnnealingLR")
    file.write(f"\nbatch_size = {batch_size}")
    # file.write(f"\nepochs = {epoch}")
    file.write(f"\nmax_stop_count = {5}")
    file.write(f"\ngrad_norm_clip = {1.0}")
    for skin, metrics in skin_metrics6.items():
        file.write(
            f"\nskin tone {skin} true label:{metrics['true']}\nskin tone {skin} predicted label:{metrics['predict']}",
        )

    for skin, metrics in skin_metrics2.items():
        file.write(
            f"\nbinary skin tone {skin} true label:{metrics['true']}\nbinary skin tone {skin} predicted label:{metrics['predict']}",
        )
