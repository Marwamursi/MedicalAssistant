
from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ---------------- Configuration ----------------
SEED = 42
IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------------- Image transforms ----------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ---------------- Load datasets ----------------
if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        "Dataset folders not found. Check data/Training and data/Testing."
    )

train_augmented = datasets.ImageFolder(
    TRAIN_DIR, transform=train_transform
)
train_plain = datasets.ImageFolder(
    TRAIN_DIR, transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    TEST_DIR, transform=eval_transform
)

if train_augmented.class_to_idx != test_dataset.class_to_idx:
    raise ValueError("Training and testing classes do not match.")

class_names = train_augmented.classes
print("Classes:", class_names)
print("Training images:", len(train_augmented))
print("Testing images:", len(test_dataset))

# Keep a separate validation subset from Training.
indices = np.arange(len(train_augmented))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.10,
    random_state=SEED,
    stratify=train_augmented.targets
)

train_dataset = Subset(train_augmented, train_idx.tolist())
val_dataset = Subset(train_plain, val_idx.tolist())

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)


# ---------------- Build model ----------------
print("\nLoading pretrained MobileNetV3-Small...")
weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Freeze the image feature extractor.
for param in model.features.parameters():
    param.requires_grad = False

# Replace the final classification layer.
input_features = model.classifier[-1].in_features
num_classes = len(class_names)
model.classifier[-1] = nn.Linear(input_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ---------------- Evaluation helper ----------------
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return (
        total_loss / total,
        correct / total,
        all_labels,
        all_predictions
    )


# ---------------- Training loop ----------------
best_val_accuracy = 0.0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    val_loss, val_accuracy, _, _ = evaluate(model, val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy or best_state is None:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "architecture": "mobilenet_v3_small"
            },
            OUTPUT_DIR / "best_model.pth"
        )

print("\nBest validation accuracy:", round(best_val_accuracy, 4))


# ---------------- Final test evaluation ----------------
checkpoint = torch.load(
    OUTPUT_DIR / "best_model.pth",
    map_location=device,
    weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_accuracy, y_true, y_pred = evaluate(
    model, test_loader
)

print("\nFinal TEST results")
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", f"{test_accuracy:.4%}")

print("\nClassification report:")
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    output_dict=True
)
print(json.dumps(report, indent=2))

print("\nConfusion matrix:")
matrix = confusion_matrix(
    y_true, y_pred, labels=list(range(len(class_names)))
)
print(matrix)

with open(OUTPUT_DIR / "evaluation_report.json", "w") as f:
    json.dump(
        {
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "class_names": class_names
        },
        f,
        indent=2
    )

print("\nModel saved to outputs/best_model.pth")
print("Report saved to outputs/evaluation_report.json")
print("Training and evaluation finished.")
