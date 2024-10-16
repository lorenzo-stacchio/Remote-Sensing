import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from transformers import CLIPModel, CLIPProcessor
# from dinov2 import DinoV2Model  # Assuming there is an available implementation

from datasets import load_dataset
# dataset = load_dataset("blanchon/FAIR1M")  # Load FAIR1M dataset
# data_loader = DataLoader(dataset['train'], batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
# for el in data_loader:
#     print(el)
#     break
# exit()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dataset and data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.COCO('path/to/coco_dataset', transform=transform)  # Example dataset
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Define function to create the model
def create_model(backbone_type='resnet', num_classes=91):
    if backbone_type == 'resnet':
        # Load pre-trained ResNet as backbone
        backbone = models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove last fully connected layers
        backbone.out_channels = 2048
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif backbone_type == 'clip':
        # Load CLIP model as backbone
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        backbone = nn.Sequential(*list(clip_model.vision_model.children())[:-2])  # Remove the classification head
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif backbone_type == 'dino':
        # Load DINO model as backbone
        dino_model = DinoV2Model.from_pretrained("facebook/dino-vit-base-patch32")
        backbone = nn.Sequential(*list(dino_model.children())[:-2])  # Remove classification layers
        model = FasterRCNN(backbone, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    return model

# Define training function
def train_model(model, data_loader, epochs=10, learning_rate=0.001):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            # During training, the model expects both the input tensors and targets (list of dictionary),
            #  containing:
            #     - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
            #       ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            #     - labels (Int64Tensor[N]): the class label for each ground-truth box
        
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader)}")

# Train different models and compare
backbone_types = ['resnet', 'clip', 'dino']
num_classes = len(dataset.coco.cats)  # Assuming using COCO dataset

for backbone_type in backbone_types:
    print(f"\nTraining with backbone: {backbone_type}")
    model = create_model(backbone_type, num_classes=num_classes)
    train_model(model, data_loader, epochs=10, learning_rate=0.001)

print("Training completed for all models.")