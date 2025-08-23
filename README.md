# Vision Transformer (ViT) Experiments

This repository contains experiments on Vision Transformer (ViT) models and hybrid approaches with CNNs, ResNets, and text-fusion methods. It includes training code, comparison results, and evaluation datasets.

---

## Folder Structure

### 1. ViT Basic
Contains baseline visual models:
- **single ViT**
- **CNN models**

The **suffix numbers (1–5)** indicate the optimizer or variant used (see *Suffix Mapping for ViT Basic*).

---

### 2. ViT ResNet
Contains **ViT-B/32 + ResNet-26** hybrid models.  
The **suffix numbers (1–3)** indicate the optimizer or variant used (see *Suffix Mapping for ViT ResNet*).

---

### 3. Desc ViT
Contains experiments where **text descriptions are fused with ViT-B/32 models**.  

- Each **subfolder name** indicates the type of description integrated.  
  Example: *Skin Tone Description* → models where **Skin Tone description** is fused.  

- Each **subfolder** has **six suffixes (1–6)** corresponding to different fusion methods (see *Fusion Mapping for Desc ViT*).

---

### 4. Root Files
- `.gitignore` → ignored files configuration  
- `Result.xlsx` → compiled results from all ViT and text-fusion experiments  
- CSV datasets → train, validation, and test splits  

---

## Suffix Mapping for ViT Basic

| Suffix | Model Variant | Optimizer / Scheduler |
|--------|---------------|-----------------------|
| **1** | ViT-B/32 | Adam optimizer + fixed learning rate |
| **2** | ViT-B/32 | SGD + momentum + fixed learning rate |
| **3** | ViT-B/32 | SGD + momentum + CosineAnnealingLR scheduler |
| **4** | ViT-B/16 | SGD + momentum + CosineAnnealingLR scheduler |
| **5** | ViT-L/32 | SGD + momentum + CosineAnnealingLR scheduler |

---

## Suffix Mapping for ViT ResNet

| Suffix | Model Variant | Optimizer / Scheduler |
|--------|---------------|-----------------------|
| **1** | ViT-B/32 + ResNet-26 | Adam optimizer + fixed learning rate |
| **2** | ViT-B/32 + ResNet-26 | SGD + momentum + fixed learning rate |
| **3** | ViT-B/32 + ResNet-26 | SGD + momentum + CosineAnnealingLR scheduler |

---

## Fusion Mapping for Desc ViT

| Suffix | Fusion Strategy |
|--------|-----------------|
| **1** | S-BERT + ViT-B/32 (Concatenation) |
| **2** | BERT + ViT-B/32 (Concatenation) |
| **3** | CLIP + ViT-B/32 (Concatenation) |
| **4** | S-BERT + ViT-B/32 (Element-wise) |
| **5** | BERT + ViT-B/32 (Element-wise) |
| **6** | CLIP + ViT-B/32 (Element-wise) |

---

## Datasets
### Fitzpatrick 17k  
- `train_data.csv` → Training set  
- `validation_data.csv` → Validation set  
- `test_data.csv` → Test set  

---

## Results

- **Result.xlsx** → compiled results from all baseline, ResNet, and text-fusion ViT experiments  

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
