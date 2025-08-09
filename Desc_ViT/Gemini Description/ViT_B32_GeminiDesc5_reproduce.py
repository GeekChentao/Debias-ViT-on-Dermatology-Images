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
import time

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

        hidden_dim = 512
        self.img_proj = nn.Sequential(
            nn.Linear(self.vit_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(self.text_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, image, text, return_attention=False):
        text_tokenized = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        text_tokenized = {k: v.to(device) for k, v in text_tokenized.items()}

        # Get attention weights from text model
        text_output = self.text_model(**text_tokenized, output_attentions=True)
        text_features = text_output.last_hidden_state[:, 0, :]  # CLS token embedding
        attention_weights = text_output.attentions  # List of attention matrices
        text_features = self.txt_proj(text_features)

        img_features = self.vit_model(image)
        img_features = self.img_proj(img_features)
        combined_features = img_features + text_features
        fused = self.fusion(combined_features)
        # combined_features = torch.cat((img_features, text_features), dim=1)

        output = self.classifier(fused)
        if return_attention:
            return output, attention_weights, text_tokenized
        return output


checkpoint = "GeminiDesc_vit32b5_skin_SGD_Momentum_0.001_CosineAnnealingLR"
checkpoint_path = f"{checkpoint}_best.pth"
output_file = f"{checkpoint}.txt"

model = VitTransformerClassifier().to(device)
skin_metrics2 = {
    i: {"total": 0, "correct": 0, "accuracy": None, "predict": list(), "true": list()}
    for i in range(1, 3)
}
skin_metrics6 = {
    i: {"total": 0, "correct": 0, "accuracy": None, "predict": list(), "true": list()}
    for i in range(1, 7)
}


def analyze_attention_importance_test(attention_weights, tokenized_text, tokenizer):
    """Analyze token importance using attention weights for test data"""
    # Average attention across all layers and heads
    avg_attention = torch.stack(attention_weights).mean(
        dim=0
    )  # [layers, batch, heads, seq_len, seq_len]
    avg_attention = avg_attention.mean(dim=(0, 2))  # Average across layers and heads

    # Get attention to CLS token (first token)
    cls_attention = avg_attention[:, 0]  # Attention from each token to CLS

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"][0])

    # Create importance dictionary
    importance = {}
    for i, (token, attention_score) in enumerate(zip(tokens, cls_attention)):
        if token != "[PAD]":
            importance[token] = attention_score.item()

    return importance


def print_token_importance(importance_dict, top_k=10):
    """Print top-k most important tokens"""
    sorted_importance = sorted(
        importance_dict.items(), key=lambda x: x[1], reverse=True
    )
    print(f"\nTop {top_k} most important tokens:")
    for i, (token, score) in enumerate(sorted_importance[:top_k]):
        print(f"{i+1}. {token}: {score:.4f}")


def log_text_importance_test(
    model,
    batch_data,
    attention_tokens_dark_correct: dict[str, int],
    attention_tokens_dark_wrong: dict[str, int],
    attention_tokens_light_correct: dict[str, int],
    attention_tokens_light_wrong: dict[str, int],
):
    """Log text importance during training"""
    images, skins, labels, descriptions = batch_data
    images = images.to(device)

    # Analyze all samples in batch
    for i in range(len(descriptions)):
        sample_image = images[i : i + 1]
        sample_text = [descriptions[i]]
        sample_label = labels[i]
        sample_skin = skins[i]

        # Attention-based importance
        with torch.no_grad():
            output, attention_weights, tokenized_text = model(
                sample_image, sample_text, return_attention=True
            )
            attention_importance = analyze_attention_importance_test(
                attention_weights, tokenized_text, model.tokenizer
            )

        # Log results for this sample
        # print(f"\nSample {i+1}/{len(descriptions)}:")
        # print(f"Text: {sample_text[0][:100]}...")  # Truncate long text
        result = output.argmax(dim=1).item()
        # print(f"True label: {sample_label}, Predicted: {result}")

        # print("Top 10 Attention-based tokens:")
        sorted_attention = sorted(
            attention_importance.items(), key=lambda x: x[1], reverse=True
        )
        for j, (token, score) in enumerate(sorted_attention[:10]):
            # print(f"  {j+1}. {token}: {score:.4f}")
            if sample_label != result:
                if sample_skin <= 4:
                    if token not in attention_tokens_light_wrong:
                        attention_tokens_light_wrong[token] = 1
                    else:
                        attention_tokens_light_wrong[token] += 1
                else:
                    if token not in attention_tokens_dark_wrong:
                        attention_tokens_dark_wrong[token] = 1
                    else:
                        attention_tokens_dark_wrong[token] += 1
            else:
                if sample_skin <= 4:
                    if token not in attention_tokens_light_correct:
                        attention_tokens_light_correct[token] = 1
                    else:
                        attention_tokens_light_correct[token] += 1
                else:
                    if token not in attention_tokens_dark_correct:
                        attention_tokens_dark_correct[token] = 1
                    else:
                        attention_tokens_dark_correct[token] += 1


def print_tokens(tokens: dict[str, int]):
    sorted_tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
    for token, score in sorted_tokens[:20]:
        print(f"{token}: {score}")


# test start
attention_tokens_dark_correct = dict()
attention_tokens_dark_wrong = dict()
attention_tokens_light_correct = dict()
attention_tokens_light_wrong = dict()

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
with torch.no_grad():
    for images, skins, labels, descriptions in test_loader:
        images = images.to(device)
        outputs = model(images, descriptions)
        outputs = outputs.argmax(dim=-1).squeeze().tolist()
        skins = skins.tolist()
        labels = labels.tolist()

        # log importance features for test data (attention only)
        batch_data = (images, skins, labels, descriptions)
        log_text_importance_test(
            model,
            batch_data,
            attention_tokens_dark_correct,
            attention_tokens_dark_wrong,
            attention_tokens_light_correct,
            attention_tokens_light_wrong,
        )

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
