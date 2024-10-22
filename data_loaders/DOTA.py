from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.curdir)
from feature_extractor import Model_type
import itertools
import copy
import tqdm, numpy as np
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.transforms import ToPILImage
from utils import set_seed 

## MANAGE BBOX AUGMENTATIONS

## https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py
## https://chatgpt.com/share/6710089f-59ac-8008-93b6-bd8b2a521eaf


# OBJECT DETECTION
## https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


class DOTAv1(Dataset):

    def _convert_image_to_rgb(self,image):
        return image.convert("RGB")

    
    def denormalize(self, image):
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return image * std + mean
    
    def set_normalize_factors(self, type_model):
        if type_model== Model_type.Clip_ViT: 
            # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
            self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        else: #type_model== "Resnet50" or "ViT" or "DinoV2"
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        
    def get_split(self,dataset_path, partition):     
        ## dataframe path          
        image_dir = dataset_path + f"{partition}/images/"
        # dataset/DOTA/FIXED/train/labelTxt-v1.0/Train_Task2_gt/trainset_reclabelTxt
        # files = [txt_file for txt_file in glob.glob(dataset_path + f"{partition}/labelTxt-v1.0/{partition.capitalize()}_Task2_gt/{partition}set_reclabelTxt/*.txt")]    
        files = [txt_file for txt_file in glob.glob(dataset_path + f"{partition}/labelTxt-v1.0/labelTxt/*.txt")]    

        # print(files)
        columns = ["image_path", "partition", "classes", "hard_objects", "boxes"]
        # print(columns)
        df = pd.DataFrame(columns=columns)
        
        # all images
        image_files =  [img_file for img_file in glob.glob(f"{image_dir}/*.png")]
        assert len(image_files) == len(files), print("DIFFERENT LENGTH", len(image_files), len(files))

        # print(len(image_files), len(files))
        dict_lists_dataframe = {x: [] for x in columns}
        # print(dict_lists_dataframe.keys())
        errors = 0
        for file_ann in tqdm.tqdm(files, total = len(files), desc = f"Building {partition} dataset"):
            # print(file_ann)
            image_path = f"{image_dir}{os.path.basename(file_ann).split('.')[0]}.png"
            # print(image_path)
            assert os.path.exists(image_path)
            
            ## basic annotations
            
            objects_ann = [line.replace("\n", "") for line in open(file_ann).readlines()]
            objects_ann = objects_ann[2:] # first two lines are metadata
            
            ## manage hard objects
            hard_objects = [int(x.split(" ")[-1]) for x in objects_ann]
            classes = [x.split(" ")[-2] for x in objects_ann]
            
            boxes = []
            for ann in objects_ann:
                x_s, y_s = [int(ann.split(" ")[idx]) for idx in [0,2,4,6]], [int(ann.split(" ")[idx]) for idx in [1,3,5,7]]
                min_x, max_x, min_y, max_y = min(x_s), max(x_s), min(y_s), max(y_s)
                boxes.append([min_x, min_y, max_x, max_y])
            
            ### CHECK FOR WRONG CLOCKWISE LABELED IMAGES AND DISCARD THEM 
            ## https://captain-whu.github.io/DOTA/dataset.html
            ## example is image P2236 which exposed a bottom-right lower than top-left
            
            if self.filter_hard_object:
                not_hard_index = [index for index, value in enumerate(hard_objects) if value == 0]
                hard_objects = [hard_objects[index] for index in not_hard_index]
                classes = [classes[index] for index in not_hard_index]
                boxes = [boxes[index] for index in not_hard_index]

            # filter boxes wrong top-left, bottom-right
            boxes_idx = [idx for idx,box in enumerate(boxes) if box[0] < box[2] and box[1] < box[3] ] 
            # print(boxes_idx)
            hard_objects = [hard_objects[index] for index in boxes_idx]
            classes = [classes[index] for index in boxes_idx]
            boxes = [boxes[index] for index in boxes_idx]
            
            if len(boxes) == 0:
                errors +=1
                # print("NON-CONSIDERED LABEL due to lenght (only 1s) or errors", image_path)
                continue
            
            # assert len(boxes) > 0, image_path continue # do not add this row
            
            dict_lists_dataframe["image_path"].append(image_path)
            dict_lists_dataframe["partition"].append(partition)
            dict_lists_dataframe["classes"].append(classes)
            dict_lists_dataframe["hard_objects"].append(hard_objects)
            dict_lists_dataframe["boxes"].append(boxes)

            # print(dict_lists_dataframe)
            # exit()
            # dict_lists_dataframe["partition"].append(partition)
            # dict_lists_dataframe["partition"].append(partition)

        
        for col in df.columns:
            df[col] = dict_lists_dataframe[col]
        ## filter hard objects 
        print(f"NON CONSIDERED ITEMS {errors}")
        return df 
        
    def __init__(self, dataset_path, split, device, model_type, filter_hard_object=True):
        assert dataset_path is not None
        self.device = device
        self.filter_hard_object = filter_hard_object
        # filter partition
        self.partition = split
        self.dataframe_csv = self.get_split(dataset_path, self.partition)
        
        self.size = 1024
        
        # classes = set()
        # for v in self.dataframe_csv["classes"]:
        #     # print(v)
        #     for cls_ in v:
        #          classes.add(cls_)
        
        self.dict_labels = {
            'soccer-ball-field': 0,
            'swimming-pool': 1,
            'tennis-court': 2,
            'ship': 3,
            'harbor': 4,
            'roundabout': 5,
            'plane': 6,
            'ground-track-field': 7,
            'storage-tank': 8,
            'small-vehicle': 9,
            'bridge': 10,
            'large-vehicle': 11,
            'basketball-court': 12,
            'helicopter': 13,
            'baseball-diamond': 14
        }
        
        self.reverse_dict_labels = {v:k for k,v in self.dict_labels.items()}
        
        print(f"Number of {self.partition}: {len(self.dataframe_csv)} with {len(self.dict_labels)} classes")
        self.set_normalize_factors(type_model=model_type)
        self.transforms = self.get_transform(self.partition)


    def get_transform(self, partition):
        transforms = []
        
        if partition == "train":
            transforms.append(T.Resize(size=(self.size,self.size), antialias=True)),
            # transforms.append(T.RandomResizedCrop(size=(self.size, self.size), antialias=True)),
            transforms.append(T.RandomHorizontalFlip(0.5)),
            transforms.append(T.RandomVerticalFlip(0.5))

        else:
            # transforms.append(T.Resize(size=(self.size, self.size), antialias=True))
            transforms.append(T.Resize(size=(self.size,self.size), antialias=True)) ## debug


        transforms.append(T.ToDtype(torch.float32, scale=True))
        transforms.append(T.ToPureTensor())
        transforms.append(T.Normalize(mean=self.mean, std=self.std))

        return T.Compose(transforms)

    def make_label(self,label):
        return self.dict_labels[label]
    
    def coarse_classes(self):
        return self.dict_labels

    def __getitem__(self, index):
        row_index = self.dataframe_csv.loc[index]
        img_path = row_index["image_path"]  # read path
        labels = row_index["classes"]  # read path
        labels = [self.dict_labels[x] for x in labels]
        boxes = row_index["boxes"]
        boxes = np.array(boxes)
        # print(boxes)
        # image = Image.open(name)
        img = read_image(img_path)
        # print("IMG SIZE", img.size())
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] =  torch.tensor(labels, dtype=torch.int64)
        target["area"] =  area
        
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        
        # try:
        #     # transform
        #     if self.partition == "train":
        #         image = self.train_transform()(image)
        #     else:  # test
        #         image = self.test_transform()(image)
        # except Exception as e:
        #     print("\n\n NAME", name)

        # data = {'image': image.to(self.device), "image_path":name, 'class': self.make_label(label)}
        
        # print("BEFORE", img_path, img.size(), target["boxes"])
        if self.transforms is not None:
            img, target_training = self.transforms(img, target)
        # print("AFTER", img_path, img.size(), target["boxes"])
        # print(target)
        # exit()

        # ## TODO: sta roba come tensore 
        # coordinates = [item for sublist in [[f"x_{i}", f"y_{i}"] for i, j in zip(range(4), range(4))] for item in sublist]
        # data.update({coord:row_index[coord] for coord in coordinates})

        return img, img_path, target, target_training

    def __len__(self):
        return len(self.dataframe_csv)

    def sample_name(self, index):
        return self.names[index]

    @property
    def label_names(self) -> List[str]:
        return self.label_setting['names']


if __name__ == "__main__":
    set_seed(42)
    
    home_dir = "/home/vrai/disk2/LorenzoStacchio/Remote Sensing/Remote Sensing/"
    dataset_path = f"{home_dir}dataset/DOTA/FIXED/"
    # training_data_config = config.TRAININGDATA
    
    train_dataset = DOTAv1(dataset_path = dataset_path, device="cpu", split="train", model_type=Model_type.Resnet50)
    train_dataloader = DataLoader(
        dataset=train_dataset, num_workers=1, shuffle=False, batch_size=1)

    test_dataset = DOTAv1(dataset_path = dataset_path, device="cpu", split="val", model_type=Model_type.Resnet50)
    test_dataloader = DataLoader(
        dataset=test_dataset, num_workers=1, shuffle=False, batch_size=1)

    out_dir = f"{home_dir}data_loaders/test_dataloaders/dotav1/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # selected_loader = test_dataloader
    # selected_dataset = test_dataset
    
    selected_loader = train_dataloader
    selected_dataset = train_dataset
    
    for idx_batch, (image, img_path, target, target_training) in enumerate(selected_loader):
        # print(image.shape)
        # print(img_path)
        # print(target)
        basename = os.path.basename(img_path[0]).split(".")[0]
        # print("TESTING", img_path[0])

        boxes = target_training["boxes"].to(torch.int32)
        area = target_training["area"].to(torch.int32)
        cls_ = target_training["labels"].to(torch.int32)
        # print(boxes)

        image_tensor = image
        image_tensor = selected_dataset.denormalize(image_tensor)
        image_tensor = image_tensor[0]
        image_pil = ToPILImage()(image_tensor)
        image_pil.save(f'{out_dir}/{basename}_{idx_batch}.jpg')
        box_np = boxes.cpu().numpy()
        area_np = area.cpu().numpy()
        cls_np = cls_.cpu().numpy()
        
        # ORIGINAL TARGET
        boxes_or_target = target["boxes"].to(torch.int32)
        cls_or_target = target["labels"].to(torch.int32)
        box_np_or_target = boxes_or_target.cpu().numpy()
        cls_np_or_target = cls_or_target.cpu().numpy()
        
        # print(box_np.shape)
        for idx_box, (box,area,clas) in enumerate(zip(box_np[0],area_np[0],cls_np[0])):
            # print(f"\n ------ BOX: {box} ORIGINAL BOX {box_np_or_target[0][idx_box]} CLASS {selected_dataset.reverse_dict_labels[clas.item()]}")
            # print(image_pil.size)
            image_pil_temp = image_pil.crop(tuple(box))
            # print(image_pil_temp.size)
            image_pil_temp.save(f'{out_dir}/{basename}_{idx_batch}_{idx_box}.jpg')
            # break
        # if idx > 10:
        
        # image_path = "dataset/DOTA/FIXED/val/images/P2718.png"
        # Original box
        # box = (1939, 2102, 2019, 2148)  # ground-track-field 0
       
        # List of tuples containing coordinates
        # coordinates = [
        #     (1939, 2102, 2020, 2105, 2019, 2148, 1939, 2144, 'ground-track-field', 0),
        #     (1070, 1344, 1102, 1348, 1095, 1398, 1063, 1394, 'ground-track-field', 0),
        #     (1986, 1641, 2026, 1636, 2040, 1710, 2000, 1717, 'ground-track-field', 0),
        #     (2606, 925, 2656, 925, 2656, 939, 2606, 939, 'bridge', 0),
        #     (2478, 1027, 2488, 1022, 2503, 1046, 2493, 1051, 'bridge', 0),
        #     (576, 1200, 615, 1200, 615, 1274, 574, 1271, 'ground-track-field', 0)
        # ]
       
        # Open the image
        image_pil = Image.open(img_path[0])
        image_pil.save(f'{out_dir}/{basename}_{idx_batch}_ORIGINAL.jpg')
        for idx_box, (box,clas) in enumerate(zip(box_np_or_target[0],cls_np_or_target[0])):
            # print(box, clas)
            image_pil_temp = image_pil.crop(tuple(box))

            # Crop the image
            # image_pil_temp = image_pil.crop(box)
            
            # Save the cropped image
            image_pil_temp.save(f'{out_dir}/{basename}_{idx_batch}_{idx_box}_ORIGINAL.jpg')
            # print(f'Cropped image {idx_box} size: {image_pil_temp.size}')
        if idx_batch > 2:
            break