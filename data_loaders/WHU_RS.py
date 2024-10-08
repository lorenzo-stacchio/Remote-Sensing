from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode
import pandas as pd
from typing import List
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
from feature_extractor import Model_type


class WHU_RS_19(Dataset):

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
            
        
    def create_df(self,dataset_path):
        # print(dataset_path)
        files = [x for x in glob.glob(dataset_path + "*/*jpg")]
        df = pd.DataFrame(columns=["image_path", "class","partition"])
        df["image_path"] = files
        df["class"] = df["image_path"].apply(lambda x: x.split("/")[-2])
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
        train_df["partition"] = ["train"] * len(train_df["partition"])
        test_df["partition"] = ["test"] * len(test_df["partition"])

        new_df = pd.DataFrame(columns=df.columns)
        new_df = pd.concat([new_df, train_df]) 
        new_df = pd.concat([new_df, test_df]) 
 
        return new_df
        
    def __init__(self, dataset_path, split, device, model_type):
        assert dataset_path is not None
        self.device = device
        
        self.dataframe_csv = self.create_df(dataset_path)

        # filter partition
        self.partition = split
        self.dataframe_csv = self.dataframe_csv[self.dataframe_csv["partition"]
                                                == self.partition]
        
        self.dataframe_csv.reset_index(drop=True, inplace=True)

        print(f"Number of {self.partition}: {len(self.dataframe_csv)}")
        self.dict_labels = {v:idx for idx,v in enumerate(sorted(self.dataframe_csv["class"].unique()))}
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
    dataset_path = "/home/vrai/Remote Sensing/dataset/DATASET WHU-RS19/"
    # training_data_config = config.TRAININGDATA
    train_dataset = WHU_RS_19(dataset_path = dataset_path, device="cuda", split="train")
    train_dataloader = DataLoader(
        dataset=train_dataset, num_workers=1, shuffle=False, batch_size=1)

    out_dir = "/home/vrai/Remote Sensing/data_loaders/test_dataloaders/"

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