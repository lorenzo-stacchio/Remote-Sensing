from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision
import torch
from PIL import Image
from torch import nn
import torch.optim as optim
# from data_loaders.dataset_utils import load_cfg_from_yaml_file
from data_loaders.RECSIS_45 import RECSIS_45
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tqdm
import warnings
warnings.filterwarnings("ignore")
import pytorch_warmup as warmup
from torchmetrics import Accuracy
from torchvision.models import ResNet50_Weights
import clip 
from feature_extractor import Model_type

def set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
    seed (int): The seed to set.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class Classifier(nn.Module):
    # NUOVO EMBEDDER INTERESSANTE: https://github.com/Kartik-3004/facexformer

    def __init__(self, embedder: Model_type, model_weight_path: str, device: str = "cuda", classes: int = 4, unfreeze:bool = False) -> None:
        super(Classifier, self).__init__()
        self.device = device  # "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder_type = embedder
        if self.embedder_type == Model_type.DinoV2:
            # LOVING YOU https://stackoverflow.com/questions/68901236/urllib-error-httperror-http-error-403-rate-limit-exceeded-when-loading-resnet1
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            self.embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device=device)
            # print(self.embedder)
            self.init_dimension = 768
        elif self.embedder_type == Model_type.Resnet50:
            self.embedder = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device=device)
            self.embedder.fc = torch.nn.Identity()
            # self.embedder.la
            self.init_dimension = 2048
        elif self.embedder_type == Model_type.ViT:
            self.embedder = torchvision.models.vit_b_16(pretrained=True).to(device=device)
            self.embedder.heads.head = torch.nn.Identity()
            self.init_dimension = 768
        elif self.embedder_type == Model_type.Clip_ViT:
            self.embedder, self.preprocess = clip.load("ViT-B/16", device=device)
            self.embedder.float() # casted to have float32 instead of float16
            self.init_dimension = 512
            
        # print(self.embedder)
        if not unfreeze:
            for param in self.embedder.parameters():
                param.requires_grad = False

        # Initialize classification head
        self.classification_head_macro = nn.Linear(
            in_features=(self.init_dimension), out_features=classes).to(self.device)

        nn.init.xavier_uniform_(self.classification_head_macro.weight)

        for param in self.classification_head_macro.parameters():
            param.requires_grad = True


    def forward(self, x):
        if self.embedder_type == Model_type.Clip_ViT:
            x = self.embedder.encode_image(x)
        else:
            x = self.embedder(x)
            
        x_macro = self.classification_head_macro(x)

        return x_macro

def parser():
    parser = argparse.ArgumentParser(description="Traing macro classes script.")

    # Add arguments
    parser.add_argument("yaml_config", help="Path Yaml Config File")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    device = "cuda:0"
    batch_size = 256
    set_seed(42)
    print("------LOADING DATASETS------")
    # dataset_path = "/home/vrai/Remote Sensing/dataset/DATASET WHU-RS19/"
    
    dataset_path = "/home/vrai/Remote Sensing/dataset/RECSIS45/"
    image_dir_path = dataset_path + "NWPU-RESISC45/"
    # embedders = [Model_type.DinoV2,Model_type.Resnet50,Model_type.ViT,Model_type.Clip_ViT]
    embedders = [Model_type.Resnet50,Model_type.ViT,Model_type.Clip_ViT,Model_type.DinoV2]
    #embedders = [Model_type.Clip_ViT,Model_type.DinoV2]
    epochs = 20

    for embedder in embedders:
       
        # training_data_config = config.TRAININGDATA
        train_dataset = RECSIS_45(dataset_path = dataset_path, image_dir_path=image_dir_path, device="cuda", split="train", model_type=embedder)
        train_dataloader = DataLoader(
            dataset=train_dataset, shuffle=False, batch_size=batch_size)

        test_dataset = RECSIS_45(dataset_path = dataset_path, image_dir_path=image_dir_path, device="cuda", split="test", model_type=embedder)
        test_dataloader = DataLoader(
            dataset=test_dataset, shuffle=False, batch_size=batch_size)

        print(f"------START OPTIMIZATION {embedder.name}------")

        exp_name = f"{embedder.name}_{epochs}"
        model = Classifier(classes=len(
            train_dataset.coarse_classes()), device=device, embedder=embedder, model_weight_path="", unfreeze=False)

        # Define a loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-5)

        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2)
        # Train the network
        # Loop over the dataset multiple times
        accuracy_metric = Accuracy(task="multiclass", num_classes=len(
            train_dataset.coarse_classes())).to(device=device)

        step = int(round(len(train_dataloader)*1))  # each epoch
        best_accuracy = float("-inf")
        log_dir = f"classification/logs/RECSIS45/{exp_name}/"
        writer = SummaryWriter(log_dir)
        set_seed(42)

        for epoch in tqdm.tqdm(range(epochs), desc="train"):

            running_loss = 0.0
            running_accuracy = 0.0
            model.train()
            for i, data in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="batch"):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, label_coarse = data["image"], data["class"]
                inputs = inputs.to(device)
                label_coarse = label_coarse.to(device)
                # print(model.device, inputs.device, label_coarse.device)
                # exit()
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, label_coarse)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                # print(outputs,label_coarse)
                # print(accuracy_metric(outputs, label_coarse))
                running_accuracy += accuracy_metric(outputs, label_coarse)

                if (i+1) % step == 0 and i > 0:    # Print every 100 mini-batches
                    actual_loss = running_loss / step
                    actual_acc = running_accuracy / step
                    print(
                        f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {actual_loss:.3f} accuracy: {actual_acc:.3f}')
                    writer.add_scalar('train_accuracy', actual_acc, epoch)
                    writer.add_scalar('train_loss_CE', actual_loss, epoch)

                    # print("Accuracy", )
                    running_loss = 0.0
                    running_accuracy = 0.0

            # with warmup_scheduler.dampening():
            # scheduler.step()

            model.eval()

            print("EVALUATION")
            test_running_accuracy = 0.0

            for i, data in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="batch"):
                inputs, label_coarse = data["image"], data["class"]
                inputs = inputs.to(device)
                label_coarse = label_coarse.to(device)
                outputs = model(inputs)
                test_running_accuracy += accuracy_metric(outputs, label_coarse)

            average_accuracy = test_running_accuracy / len(test_dataloader)
            print(
                f'\n[Epoch {epoch + 1}, batch {i + 1}]  test accuracy: {average_accuracy:.3f}')
            writer.add_scalar('test_accuracy', average_accuracy, epoch+1)

            if average_accuracy > best_accuracy:
                torch.save(model.state_dict(
                ), f"classification/my_weights/best_model_lr_higher_{exp_name}_{epochs}.pth")
                best_accuracy = average_accuracy

        print("best_accuracy", best_accuracy)
        print('Finished Training')
        writer.close()
