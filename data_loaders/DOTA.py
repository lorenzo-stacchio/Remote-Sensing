from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from typing import List
from torchvision import transforms
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
import tqdm 
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

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
        files = [txt_file for txt_file in glob.glob(dataset_path + f"{partition}/labelTxt-v1.0/{partition.capitalize()}_Task2_gt/{partition}set_reclabelTxt/*.txt")]    
        # print(files)
        columns = ["image_path", "partition"]
        coordinates = [item for sublist in [[f"x_{i}", f"y_{i}"] for i, j in zip(range(4), range(4))] for item in sublist]
        ann_coordinates = copy.deepcopy(coordinates)
        ann_coordinates.extend(["class","hard_object"])
        
        columns.extend(ann_coordinates)
        # print(columns)
        df = pd.DataFrame(columns=columns)
        
        # all images
        image_files =  [img_file for img_file in glob.glob(f"{image_dir}/*.png")]   
        assert len(image_files) == len(files), print("DIFFERENT LENGTH", len(image_files), len(files))

        # print(len(image_files), len(files)) 
        dict_lists_dataframe = {"image_path": [], "partition": []}
        dict_lists_dataframe.update({index:[] for index in ann_coordinates})
        print(dict_lists_dataframe.keys())

        for file_ann in tqdm.tqdm(files, total = len(files), desc = f"Building {partition} dataset"):
            # print(file_ann)
            image_path = f"{image_dir}{os.path.basename(file_ann).split('.')[0]}.png"
            # print(image_path)
            assert os.path.exists(image_path)
            objects_ann = [line.replace("\n", "") for line in open(file_ann).readlines()]
            for obj_ann in objects_ann:
                dict_lists_dataframe["image_path"].append(image_path)
                dict_lists_dataframe["partition"].append(partition)
                ## coordinates
                for index, value in zip(ann_coordinates, obj_ann.split(" ")): 
                    dict_lists_dataframe[index].append(value)
        
        for col in df.columns:
            df[col] = dict_lists_dataframe[col]
        
        # float to int for 
        for c in coordinates:
            df[c] = df[c].astype(float)

        # print(len(df))
        # print(df.head(5))                
        return df 
        
    def __init__(self, dataset_path, split, device, model_type):
        assert dataset_path is not None
        self.device = device
        
        # filter partition
        self.partition = split
        self.dataframe_csv = self.get_split(dataset_path, self.partition)
        
        self.size = 512
        # print(list(self.dataframe_csv["image_path"][:10]))
        # print(self.dataframe_csv.head(10))
        
        self.dict_labels = {v:idx for idx,v in enumerate(sorted(self.dataframe_csv["class"].unique()))}
        print(f"Number of {self.partition}: {len(self.dataframe_csv)} with {len(self.dict_labels)} classes")
        self.set_normalize_factors(type_model=model_type)
        # exit()

    def train_transform(self):
        return transforms.Compose([
            self._convert_image_to_rgb,
            transforms.Resize((self.size,self.size) interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomCrop(self.size),
            transforms.ToTensor(),
            # Normalize with mean and std of imagenet
            transforms.Normalize(self.mean,self.std)
        ])

    def test_transform(self):
        return transforms.Compose([
        self._convert_image_to_rgb,
        transforms.Resize((self.size,self.size), interpolation=InterpolationMode.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalize with mean and std of imagenet
        transforms.Normalize(self.mean,self.std)
    ])

    def make_label(self,label):
        return self.dict_labels[label]
    
    def coarse_classes(self):
        return self.dict_labels

    def __getitem__(self, index):
        row_index = self.dataframe_csv.loc[index]
        name = row_index["image_path"]  # read path
        label = row_index["class"]  # read path
        image = Image.open(name)
        try:
            # transform
            if self.partition == "train":
                image = self.train_transform()(image)
            else:  # test
                image = self.test_transform()(image)
        except Exception as e:
            print("\n\n NAME", name)

        data = {'image': image.to(self.device), "image_path":name, 'class': self.make_label(label)}
        
        ## TODO: sta roba come tensore 
        coordinates = [item for sublist in [[f"x_{i}", f"y_{i}"] for i, j in zip(range(4), range(4))] for item in sublist]
        data.update({coord:row_index[coord] for coord in coordinates})

        return data

    def __len__(self):
        return len(self.dataframe_csv)

    def sample_name(self, index):
        return self.names[index]

    @property
    def label_names(self) -> List[str]:
        return self.label_setting['names']


if __name__ == "__main__":
    home_dir = "/home/vrai/disk2/LorenzoStacchio/Remote Sensing/Remote Sensing/"
    dataset_path = f"{home_dir}dataset/DOTA/FIXED/"
    # training_data_config = config.TRAININGDATA
    train_dataset = DOTAv1(dataset_path = dataset_path, device="cpu", split="train", model_type=Model_type.Resnet50)
    train_dataloader = DataLoader(
        dataset=train_dataset, num_workers=1, shuffle=False, batch_size=1)

    test_dataset = DOTAv1(dataset_path = dataset_path, device="cpu", split="val", model_type=Model_type.Resnet50)
    test_dataloader = DataLoader(
        dataset=test_dataset, num_workers=1, shuffle=False, batch_size=1)

    # test_dataset = RECSIS_45(dataset_path = dataset_path, image_dir_path=image_dir_path, device="cpu", split="test", model_type=Model_type.Resnet50)
    # test_dataloader = DataLoader(
    #     dataset=test_dataset, num_workers=1, shuffle=False, batch_size=1)

    out_dir = f"{home_dir}data_loaders/test_dataloaders/dotav1/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for i, data in enumerate(train_dataloader):
        print(data.keys())
        print(i, data["image_path"], data["class"])
        x_0 = int(data['x_0'])
        y_0 = int(data['y_0'])
        x_1 = int(data['x_1'])
        y_1 = int(data['y_1'])
        x_2 = int(data['x_2'])
        y_2 = int(data['y_2'])
        x_3 = int(data['x_3'])
        y_3 = int(data['y_3'])        
        print(x_0, y_0, x_2, y_2) # anti
        image_tensor = data["image"]
        image_tensor = train_dataset.denormalize(image_tensor)
        image_tensor = image_tensor[0]
        image_pil = transforms.ToPILImage()(image_tensor)
        # image_pil = image_pil.crop((x_0, y_0, x_2, y_2))

        image_pil.save(f'{out_dir}/image{i}.jpg')
        if i > 10:
            break