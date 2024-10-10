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


# DATASET FROM
# https://huggingface.co/datasets/timm/resisc45
# https://www.tensorflow.org/datasets/catalog/resisc45?hl=it

# SPLITS FROM
# https://github.com/google-research/google-research/blob/master/remote_sensing_representations/README.md#dataset-splits


class RECSIS_45(Dataset):

    def _convert_image_to_rgb(self,image):
        return image.convert("RGB")

    
    def denormalize(self, image):
        # mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)

        return image * std + mean
    
    def set_normalize_factors(self, type_model):
        if type_model== Model_type.Clip_ViT: 
            # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
            self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        else: #type_model== "Resnet50" or "ViT" or "DinoV2"
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            
        
    def get_split(self,dataset_path, image_dir_path, partition):
        files = [x.replace("\n","") for x in open(dataset_path + f"resisc45-{partition}.txt").readlines()]        
        df = pd.DataFrame(columns=["image_path", "class","partition"])
        df["image_path"] = files
        df["class"] = df["image_path"].apply(lambda x: "_".join(x.split("_")[:-1]))
        df["image_path"] = df.apply(lambda x: f"{image_dir_path}{x['class']}/{x['image_path']}", axis=1)
        df["partition"] = [partition] * len(df)
        return df
        
    def __init__(self, dataset_path, split, image_dir_path, device, model_type):
        assert dataset_path is not None
        self.device = device
        
        # filter partition
        self.partition = split
        self.dataframe_csv = self.get_split(dataset_path, image_dir_path, self.partition)
        # print(list(self.dataframe_csv["image_path"][:10]))
        # print(self.dataframe_csv.head(10))
        
        self.dict_labels = {v:idx for idx,v in enumerate(sorted(self.dataframe_csv["class"].unique()))}
        print(f"Number of {self.partition}: {len(self.dataframe_csv)} with {len(self.dict_labels)} classes")
        # print(self.dict_labels)

        self.set_normalize_factors(type_model=model_type)
        
        
    def train_transform(self):
        return transforms.Compose([
            self._convert_image_to_rgb,
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            # Normalize with mean and std of imagenet
            transforms.Normalize(self.mean,self.std)
        ])

    def test_transform(self):
        return transforms.Compose([
        self._convert_image_to_rgb,
        transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
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

        return data

    def __len__(self):
        return len(self.dataframe_csv)

    def sample_name(self, index):
        return self.names[index]

    @property
    def label_names(self) -> List[str]:
        return self.label_setting['names']


if __name__ == "__main__":
    dataset_path = "/home/vrai/Remote Sensing/dataset/RECSIS45/"
    image_dir_path = dataset_path + "NWPU-RESISC45/"
    # training_data_config = config.TRAININGDATA
    train_dataset = RECSIS_45(dataset_path = dataset_path, image_dir_path=image_dir_path, device="cpu", split="train", model_type=Model_type.Resnet50)
    train_dataloader = DataLoader(
        dataset=train_dataset, num_workers=1, shuffle=False, batch_size=1)


    test_dataset = RECSIS_45(dataset_path = dataset_path, image_dir_path=image_dir_path, device="cpu", split="test", model_type=Model_type.Resnet50)
    test_dataloader = DataLoader(
        dataset=test_dataset, num_workers=1, shuffle=False, batch_size=1)

    out_dir = "/home/vrai/Remote Sensing/data_loaders/test_dataloaders/recsis45/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for i, data in enumerate(train_dataloader):
        print(i, data["image_path"], data["class"])

        image_tensor = data["image"]
        image_tensor = train_dataset.denormalize(image_tensor)
        image_tensor = image_tensor[0]
        to_pil = transforms.ToPILImage()
        image_pil = to_pil(image_tensor)
        image_pil.save(f'{out_dir}/image{i}.jpg')
        if i > 10:
            break