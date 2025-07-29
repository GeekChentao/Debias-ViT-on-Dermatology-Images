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
import clip

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_clip, preprocess = clip.load("ViT-B/32", device)

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

        description = self.df.iloc[idx]["MONET Full Description"]
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


class ClipClassifier(nn.Module):
    def __init__(self):
        super(ClipClassifier, self).__init__()
        self.hidden_dim = 512

        self.img_proj = nn.Sequential(
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(self.hidden_dim, 2)

    def forward(self, image, text):
        max_words = 20
        text = tuple(" ".join(t.split()[:max_words]) for t in text)

        # Ensure text is a list of strings
        if isinstance(text, tuple):
            text = list(text)  # Convert tuple to list
        elif isinstance(text, str):
            text = [text]  # Convert single string to list
        with torch.no_grad():
            img_features = model_clip.encode_image(image.to(device)).float()
            text_tokens = clip.tokenize(text).to(device)
            text_features = model_clip.encode_text(text_tokens).float()

        img_features = self.img_proj(img_features)
        text_features = self.txt_proj(text_features)
        combined_features = img_features + text_features
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        return output


checkpoint = "MONET_fulldesc_vit32b6_skin_SGD_Momentum_0.001_CosineAnnealingLR"
checkpoint_path = f"{checkpoint}_best.pth"
output_file = f"{checkpoint}.txt"

model = ClipClassifier().to(device)
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
    file.write(f"\ntokenizer = CLIP")
    file.write(f"\nintegrate_way = fusion")
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
