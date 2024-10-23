import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
# from transformers import CLIPModel, CLIPProcessor
# from dinov2 import DinoV2Model  # Assuming there is an available implementation
from feature_extractor import Model_type
from datasets import load_dataset
from data_loaders.DOTA import DOTAv1
from utils import set_seed 
from data_loaders.utils import collate_fn 
import tqdm 
from torchvision.ops import box_iou
import clip 
import timm

# Function to save model weights
def save_model_weights(model, epoch, precision, model_path):
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}_precision_{precision:.4f}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    
def evaluate_model(model, data_loader, device):
    model.eval()
    results = []
    gt_boxes = []
    
    with torch.no_grad():
        for images, _, targets in tqdm.tqdm(data_loader):
            images = list(image.to(device) for image in images)
            outputs = model(images)  # Outputs are a list of dicts with 'boxes' and 'labels'
            
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                gt_box = targets[i]['boxes'].cpu()
                
                # Store results for mAP or IoU calculation later
                results.append({
                    'pred_boxes': pred_boxes,
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu(),
                })
                gt_boxes.append(gt_box)
    
    # Perform IoU and mAP calculations
    return calculate_map(results, gt_boxes)

def calculate_map(results, gt_boxes, iou_threshold=0.5):
    """Calculate mAP at a given IoU threshold."""
    total_correct = 0
    total_pred = 0
    
    for i, res in enumerate(results):
        pred_boxes = res['pred_boxes']
        scores = res['scores']
        gt_box = gt_boxes[i]
        
        if len(pred_boxes) == 0 or len(gt_box) == 0:
            continue
        
        # Calculate IoU between predicted and ground-truth boxes
        iou = box_iou(pred_boxes, gt_box)
        
        # Count correct predictions above IoU threshold
        correct_pred = (iou > iou_threshold).sum().item()
        total_correct += correct_pred
        total_pred += len(pred_boxes)
    
    # Calculate Precision and Average Precision
    precision = total_correct / total_pred if total_pred > 0 else 0
    return precision


class ViTBackbone(nn.Module):
    def __init__(self, type_vit):
        super(ViTBackbone, self).__init__()
        # Load the ViT model from timm with pre-trained weights
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, features_only=True)
        self.type_vit = type_vit
        print("type_vit", self.type_vit)
        self.vit = timm.create_model(self.type_vit, pretrained=True, num_classes=0)
        self.patches = 14
        self.out_channels = 768  # For vit_base_patch16_224, the token size is 768

    def forward(self, x):
        # Get all the features from ViT
        # features = self.vit(x)
        features = self.vit.forward_features(x)
        # print(features.shape)
        ## remove class token
        features = features[:, 1:, :] 
        # reshape by patches
        # print("VIT FEATURES ", features.shape)
        if self.type_vit == "vit_base_patch32_clip_448.laion2b_ft_in12k_in1k":
            # features = features.reshape(features.shape[0], 14, 14, 768)
            features = features.reshape(features.shape[0], 14, 14, 768)
        elif self.type_vit == "vit_base_patch14_dinov2":
            features = features.reshape(features.shape[0], 37, 37, 768)
        elif self.type_vit == "vit_base_patch14_reg4_dinov2":
            features = features.reshape(features.shape[0], 37, 37, 768)
        elif self.type_vit == "vit_base_patch32_384":
            features = features.reshape(features.shape[0], 12, 12, 768)
        features = features.permute(0, 3, 1, 2)

        return features


class DinoBackbone(nn.Module):
    def __init__(self, type_dino):
        super(DinoBackbone, self).__init__()
        # Load the ViT model from timm with pre-trained weights
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, features_only=True)
        self.type_dino = type_dino
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.dino = torch.hub.load('facebookresearch/dinov2', self.type_dino).to(device=device)            
        # print(self.dino)
        
    def forward(self, x):
        # Get all the features from ViT
        # features = self.vit(x)
        # print("iMAGE SHAPE DINO", x.shape)
        # if 544 in list(x.shape):
        #     print(x.shape)
        #     exit()

        features_total = self.dino.forward_features(x)
        ## possible outs: "x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm", "masks"
        # print(len(features))
        # for el in features:
        #     print(el.shape)
        features = features_total["x_norm_patchtokens"]
        # print(patch_tokens.shape)
        ## remove class token
        # features = features[:, 1:, :] 
        # reshape by patches
        # print("dino FEATURES ", features.shape)
        if self.type_dino == "dinov2_vitb14":
            features = features.reshape(features.shape[0], 37, 37, 768)
        features = features.permute(0, 3, 1, 2)

        return features
    
class Detector(nn.Module):
    # NUOVO EMBEDDER INTERESSANTE: https://github.com/Kartik-3004/facexformer

    def __init__(self, embedder: Model_type, model_weight_path: str, classes: int, device: str = "cuda", unfreeze:bool = False) -> None:
        super(Detector, self).__init__()
        self.device = device  # "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder_type = embedder
        self.image_size = 0
        self.size_divisible  = 0
        if self.embedder_type == Model_type.Resnet50:
            self.embedder = models.resnet50(pretrained=True)
            print(self.embedder)
            self.embedder = nn.Sequential(*list(self.embedder.children())[:-2])  # Remove last fully connected layers
            self.embedder.out_channels = 2048
            self.image_size = 448
            self.size_divisible = 32 # default r-cnn
            # self.embedder.la
            # self.init_dimension = 2048
        # elif self.embedder_type == Model_type.ViT:
        #     self.embedder = torchvision.models.vit_b_16(pretrained=True).to(device=device)
        #     self.embedder.heads.head = torch.nn.Identity()
        #     # self.init_dimension = 768
        elif self.embedder_type == Model_type.Clip_ViT:
            # find all of them here: https://huggingface.co/timm
            ## facciamolo funzionare cazzo ha sia vit che clip vit
            ## https://github.com/rwightman/timm/blob/b3f3a005a0eaf9326e1dab19dbbc4a947863da4e/timm/models/vision_transformer.py
            self.embedder = ViTBackbone(type_vit = "vit_base_patch32_clip_448.laion2b_ft_in12k_in1k")
            # self.embedder = timm.create_model('vit_base_patch16_224', pretrained=True, features_only=True)
            # print(type(self.embedder))
            # self.embedder, self.preprocess = clip.load("ViT-B/32")
            # #self.embedder = nn.Sequential(*list(self.embedder.visual.children())[:-1])  # Remove the classification head
            # self.embedder = self.embedder.visual
            # self.embedder.proj == None
            # self.embedder.ln_post == nn.Identity()
            # print(self.embedder.transformer.weight.dtype)
            # self.embedder.float() # casted to have float32 instead of float16
            self.embedder.out_channels = 768
            self.image_size = 448
            self.size_divisible = 32 # default r-cnn

            # self.init_dimension = 512
        elif self.embedder_type == Model_type.DinoV2:
            # self.embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            # LOVING YOU https://stackoverflow.com/questions/68901236/urllib-error-httperror-http-error-403-rate-limit-exceeded-when-loading-resnet1
            # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            # self.embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device=device)            
            self.embedder = DinoBackbone("dinov2_vitb14")
            self.embedder.out_channels = 768
            self.image_size = 518
            # exception due to image size
            self.size_divisible = 37
           
            # print(self.embedder)
        elif self.embedder_type == Model_type.ViT:
            # self.embedder = ViTBackbone(type_vit = "hf_hub:timm/vit_large_patch14_dinov2.lvd142m")
            self.embedder = ViTBackbone(type_vit = "vit_base_patch32_384")
            self.image_size = 384
            self.embedder.out_channels = 768
            self.size_divisible = 32 # default r-cnn


        # print(self.embedder)
        print("------- SANITY CHECK -------")
        random_image_tensor = torch.rand(1, 3, self.image_size, self.image_size, dtype=torch.float32)#.to("cuda")
        # print(random_image_tensor.dtype)        
        random_image_tensor = random_image_tensor.to(self.device)
        self.embedder = self.embedder.to(self.device) 
        out = self.embedder(random_image_tensor)
        print("SHAPE FOR RCNN", out.shape)
        print("------- END SANITY CHECK -------")
        # exit()
        
        for param in self.embedder.parameters():
            param.requires_grad = False

        self.anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
        )
       
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # print("IMAGE SIZE BACKBONE", self.image_size)
        
        self.model = FasterRCNN(self.embedder, num_classes=classes,  
                           rpn_anchor_generator=self.anchor_generator,
                           box_roi_pool=self.roi_pooler, min_size=self.image_size, max_size = self.image_size,
                           size_divisible = self.size_divisible)
        
        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

    def forward(self, images, targets = None):
        print("\n\nSHAPE IN FIRST STEP DETECTOR", images[0].shape, "\n\n")
        
        if targets:
            return self.model(images,targets)
        else:
            return self.model(images)



# Define function to create the model
def create_model(backbone_type, num_classes):
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
        )
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    if backbone_type == 'resnet':
        # Load pre-trained ResNet as backbone
        backbone = models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove last fully connected layers
        backbone.out_channels = 2048
        model = FasterRCNN(backbone, num_classes=num_classes,  
                           rpn_anchor_generator=anchor_generator,
                            box_roi_pool=roi_pooler)
    elif backbone_type == 'clip':
        # Load CLIP model as backbone
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        backbone = nn.Sequential(*list(clip_model.vision_model.children())[:-2])  # Remove the classification head
        model = FasterRCNN(backbone, num_classes=num_classes,  
                           rpn_anchor_generator=anchor_generator,
                            box_roi_pool=roi_pooler)
    elif backbone_type == 'dino':
        # Load DINO model as backbone
        dino_model = DinoV2Model.from_pretrained("facebook/dino-vit-base-patch32")
        backbone = nn.Sequential(*list(dino_model.children())[:-2])  # Remove classification layers
        model = FasterRCNN(backbone, num_classes=num_classes,  
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")
    ## check head faster RCNN: 
    ## https://pytorch.org/vision/main/_modules/torchvision/models/detection/faster_rcnn.html#:~:text=class%20FasterRCNN(GeneralizedRCNN)%3A
    
    return model

# Define training function
def train_model(model, data_loader, test_dataloader, epochs, embedder_name):
    exp_name = f"{embedder_name}_{epochs}"
    log_dir = f"classification/logs_objd/{exp_name}/"
    writer = SummaryWriter(log_dir)
    set_seed(42)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    best_precision = 0  # Initialize best precision

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for images, _, targets in tqdm.tqdm(data_loader):
            # During training, the model expects both the input tensors and targets (list of dictionary),
            #  containing:
            #     - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
            #       ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            #     - labels (Int64Tensor[N]): the class label for each ground-truth box
        
            images = list(image.to(device) for image in images)
            # for image in images:
            #     print(image.shape)
            
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # exit()
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
        writer.add_scalar('train_loss', epoch_loss/len(data_loader), epoch+1)

        if epoch%5 == 0: 
            # TODO: implement evaluation for Object detection task
            model.eval()
            print("Evaluating on the test dataset...")
            precision = evaluate_model(model, test_dataloader, device)
            writer.add_scalar('test_precision', precision, epoch+1)

            print(f"Precision at epoch {epoch + 1}: {precision}")
            if precision > best_precision:
                print(f"Precision improved from {best_precision} to {precision}")
                best_precision = precision
                save_model_weights(model, epoch, precision, f"{home_dir}/classification/weights_objd/resnet50.pth")
            else:
                print(f"Precision did not improve, still {best_precision}")

            
if __name__ == "__main__":
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cuda"

    set_seed(42)
        
    home_dir = "/home/vrai/disk2/LorenzoStacchio/Remote Sensing/Remote Sensing/"
    dataset_path = f"{home_dir}dataset/DOTA/FIXED/"
    # training_data_config = config.TRAININGDATA

    # Train different models and compare
    # backbone_types = [Model_type.Resnet50]
    backbone_types = [Model_type.DinoV2]
    # backbone_types = [Model_type.ViT]
    # backbone_types = [Model_type.Clip_ViT]
    # backbone_types = ['resnet', 'clip', 'dino']

    num_classes = 15  # make a json and load it
    epochs = 3
    for backbone_type in backbone_types:
        print(f"\nTraining with backbone: {backbone_type}")
        # model = create_model(backbone_type, num_classes=num_classes)
        model = Detector(embedder= backbone_type, model_weight_path = "", 
                        device=device, classes = num_classes)
        
        batch_size =  8

        train_dataset = DOTAv1(dataset_path = dataset_path, image_size = model.image_size,
                            device=device, split="train", model_type=Model_type.Resnet50)
        
        train_dataloader = DataLoader(
            dataset=train_dataset, num_workers=1, shuffle=True, batch_size=batch_size, collate_fn = collate_fn)

        test_dataset = DOTAv1(dataset_path = dataset_path, image_size = model.image_size,
                            device=device, split="val", model_type=Model_type.Resnet50)
        test_dataloader = DataLoader(
            dataset=test_dataset, num_workers=1, shuffle=True, batch_size=batch_size, collate_fn = collate_fn)

        train_model(model, train_dataloader, test_dataloader, epochs=epochs,  embedder_name=backbone_type.name)

    print("Training completed for all models.")