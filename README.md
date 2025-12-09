# stats507_final
A deep learning pipeline for face-shape classification, hairstyle retrieval, and GAN-based hairstyle synthesis.

# Face Shape Classification, Similarity-Based Hairstyle Recommendation & GAN Hairstyle Transfer

This repository implements an end-to-end system that:

1. **Classifies face shapes** using an EfficientNet-B5 model.  
2. **Recommends hairstyles** by finding visually similar faces within a labeled dataset.  
3. **Applies hairstyles** onto the original face using a StyleGAN2-based editing pipeline (Barbershop).

The project is split into:

- `main.ipynb` — data preparation, model training, evaluation, and hairstyle recommendation logic.  
- `Barbershop-main/` — GAN-based hairstyle transfer module, including:
  - `Barbershop-main/align_face.py`
  - `Barbershop-main/main.py`

---

## 1. Project Overview

Given an input face image, the system performs:

1. **Face shape classification** → predicts one of five categories:  
   `Heart`, `Oblong`, `Oval`, `Round`, `Square`.

2. **Similarity-based hairstyle recommendation** →  
   finds Top-3 most visually similar faces within the same face-shape category and uses their hairstyles as recommendations.

3. **GAN hairstyle transfer (Barbershop)** →  
   applies the recommended hairstyle(s) onto the original face image using a StyleGAN2-based latent-space editing framework.

---

## 2. Directory Structure

A typical layout of this repository:

```text
project_root/
│
├── main.ipynb                        # Main notebook: training, evaluation, recommendation
│
├── Barbershop-main/                  # GAN hairstyle transfer module
│   ├── main.py                       # Barbershop pipeline entry point
│   ├── align_face.py                 # Face alignment using MediaPipe
│   ├── models/
│   │   ├── Embedding.py              # Inversion to StyleGAN latent space
│   │   ├── Alignment.py              # Latent alignment / warping utilities
│   │   ├── Blending.py               # Loss functions and blending optimization
│   │   └── ...
│   ├── pretrained_models/
│   │   └── ffhq.pt                   # Style2 FFHQ pretrained weights
│   ├── unprocessed/                  # Input images before alignment (original + hairstyle samples)
│   ├── input/                        # Aligned faces for GAN processing
│   │   └── face/                     # Typically used as the main aligned input folder
│   ├── output/                       # Final edited images (hairstyle-applied results)
│   └── ...
│
├── FaceShape Dataset/                # Dataset used for face-shape classification
│   ├── training_set/
│   │   ├── Heart/
│   │   ├── Oblong/
│   │   ├── Oval/
│   │   ├── Round/
│   │   └── Square/
│   └── testing_set/
│       ├── Heart/
│       ├── Oblong/
│       ├── Oval/
│       ├── Round/
│       └── Square/
│
└── README.md
```
## Pretrained Model

Due to GitHub's 100MB file size limit, the pretrained EfficientNet model `best_model.pth` is not included in this repository.
You can download it from the following link and place it in the project root:

- Download: https://drive.google.com/file/d/17uwtzFxx5Vb4rQ6mZGae9Kq-otqjAxhQ/view?usp=sharing
- Target path: `./best_model.pth`

## 3. Environment & Requirements

### 3.1. Recommended Environment

- Python **3.8+**
- CUDA-capable **GPU** (strongly recommended for GAN / Barbershop)
- Operating System: **Linux** or compatible environment (Google Colab recommended)

### 3.2. Core Dependencies

From `main.ipynb`, `Barbershop-main/main.py`, and `Barbershop-main/align_face.py`, the following libraries are required:

#### Deep Learning / ML
- `torch`
- `torchvision`
- `efficientnet_pytorch`
- `scikit-learn`

#### Image Processing
- `opencv-python` (`cv2`)
- `Pillow` (`PIL`)
- `mediapipe`
- `matplotlib`

#### Utilities
- `numpy`
- `pandas`
- `tqdm`

---
## 4. Dataset: Face Shape Classification

The notebook main.ipynb expects a dataset directory structured as:

FaceShape Dataset/


Inside, images are organized into training_set/ and testing_set/, each containing subfolders per face-shape class:

FaceShape Dataset/
├── training_set/
│   ├── Heart/
│   ├── Oblong/
│   ├── Oval/
│   ├── Round/
│   └── Square/
└── testing_set/
    ├── Heart/
    ├── Oblong/
    ├── Oval/
    ├── Round/
    └── Square/


Each subfolder contains .jpg or .png images labeled by folder name.

The notebook automatically:

Walks the dataset directory

Builds train, validation, and test splits into pandas.DataFrames

Maps labels to integer indices:

['Heart', 'Oblong', 'Oval', 'Round', 'Square']

## 5. Face Shape Classification (EfficientNet-B5)

All training and evaluation logic is implemented in main.ipynb.

### 5.1. Configuration

A simple configuration class:

class args:
    data_dir = "FaceShape Dataset"
    batch_size = 10
    n_epochs = 20
    learning_rate = 0.001
    debug = False  # trains on a tiny subset if True

### 5.2. Dataset & DataLoader

A custom FaceShapeDataset is used:

Reads image paths & labels from a DataFrame

Loads images with:

PIL.Image.open(path).convert('L')


Applies transforms:

Resize(256), CenterCrop(224), ToTensor()


Three DataLoaders:

train_loader

val_loader

test_loader

with batch_size = args.batch_size.

### 5.3. Model: EfficientNet-B5

Model definition:

class EffNet(nn.Module):
    def __init__(self, num_classes=5):
        super(EffNet, self).__init__()
        self.eff = EfficientNet.from_pretrained(
            'efficientnet-b5',
            num_classes=num_classes,
            in_channels=1  # grayscale input
        )

    def forward(self, x):
        return self.eff(x)


Key notes:

Uses pretrained EfficientNet-B5

Modified to accept 1-channel grayscale

Outputs 5 class logits

### 5.4. Training Procedure

The notebook:

Moves model to CUDA if available:

device = "cuda:0" if torch.cuda.is_available() else "cpu"


Uses:

Loss: nn.CrossEntropyLoss

Optimizer: torch.optim.Adam

Training loop includes:

Forward pass

Backward pass

Optimizer step

Validation accuracy computation

Best model weights saved as:

torch.save(model.state_dict(), "best_model.pth")

### 5.5. Evaluation

Load best model:

model = EffNet().to(device)
model.load_state_dict(torch.load("best_model.pth"))


On test set:

Computes accuracy

Computes average loss

Generates a classification report using:

sklearn.metrics.classification_report

## 6. Face-Shape Prediction & Hairstyle Recommendation

After training, the notebook:

Predicts face shapes

Recommends Top-3 hairstyles using similarity

### 6.1. Face Detection & Cropping (MediaPipe)

Process:

Convert BGR → RGB
Run MediaPipe detection:
model_selection=1
min_detection_confidence=0.5

Crop bounding box if detected

### 6.2. Feature Extraction (Color Histogram)

Steps:

Resize cropped face to 224×224
Compute a 3D RGB histogram: 8 bins per channel → 8 × 8 × 8 = 512 features
Normalize histogram
Flatten into vector

### 6.3. Similarity Matching

Procedure:

Filter candidate hairstyles by predicted face-shape label

For each candidate:
a. Load image
b. Crop face
c. Compute histogram vector
d. Compute Euclidean distance
e. Sort ascending
f. Select Top-3
Save output images to: Barbershop-main/unprocessed/


Filenames:

original_pic_<id>.jpeg
hair0_<id>.jpeg
hair1_<id>.jpeg
hair2_<id>.jpeg

## 7. GAN Hairstyle Transfer (Barbershop)

Implemented in:

Barbershop-main/align_face.py
Barbershop-main/main.py

### 7.1. Face Alignment (align_face.py)

Reads from: Barbershop-main/unprocessed/

Uses MediaPipe to:
a. Detect face
b. Crop and align
c. Optionally resize (output_size=512)

Saves into: Barbershop-main/input/face/

Example:

cd Barbershop-main
python align_face.py \
  -unprocessed_dir unprocessed \
  -aligned_dir input/face \
  -output_size 512

### 7.2. Hairstyle Transfer (main.py)

main.py does:

a. Parse CLI arguments
b. Invert images into StyleGAN latent space
c. Perform feature alignment
d. Blend identity + hairstyle

Save results into: Barbershop-main/output/


Example command:

python main.py \
  --im_path1 input/face/original_pic_742.png \
  --im_path2 input/face/hair0_742.png \
  --im_path3 input/face/hair0_742.png \
  --sign realistic \
  --smooth 5 \
  --device cuda

## 8. End-to-End Usage
### Step 1 — Prepare Dataset
Place images into correct folders under: FaceShape Dataset/

### Step 2 — Train Classifier
Run main.ipynb
Confirm: best_model.pth is saved.

### Step 3 — Predict & Recommend Hairstyles
Run the recommendation section in notebook.
Outputs are saved into: Barbershop-main/unprocessed/

### Step 4 — Align Faces
cd Barbershop-main
python align_face.py -unprocessed_dir unprocessed -aligned_dir input/face -output_size 512

### Step 5 — Apply Hairstyle (GAN)
python main.py --im_path1 input/face/original_pic.png ...
Final results appear in: output/

## 9. Limitations & Future Improvements

a. MediaPipe detection can fail on extreme angles / occlusion
b. Histogram similarity is simple → could replace with deep embeddings (ArcFace, etc.)
c. GAN editing is computationally heavy → GPU required
d. Performance varies based on dataset diversity

