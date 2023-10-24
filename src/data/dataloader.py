import numpy as np
import random
import torch
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T


NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def to_device(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def denormalize(tensor):
    return tensor * 0.225 + 0.45

class ImageDataset(torch.utils.data.DataLoader):
    def __init__(self):
        self.path = 'data/processed/Food Images'
        self.image_files = os.listdir(self.path)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.image_files[idx])
        image = Image.open(image_path)
        image = self.transform(image)
        return image

class ContrastiveImageDataset(torch.utils.data.DataLoader):
    def __init__(self):
        self.path = 'data/processed/Food Images'
        self.image_files = os.listdir(self.path)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.image_files)
    def random_idx(self, idx):
        r_idx =  torch.randint(0,len(self.image_files),(1,)).item()
        if r_idx == idx:
            r_idx = self.random_idx(idx)
        return r_idx
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.image_files[idx])
        image = Image.open(image_path)
        r_idx = self.random_idx(idx)
        contrast_img = os.path.join(self.path, self.image_files[r_idx])
        contrast_img = Image.open(contrast_img)
        return self.transform(image), self.transform(contrast_img)
    
class TextDataSet(torch.utils.data.DataLoader):
    def __init__(self):
        FILE_PATH = 'data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
        self.csv = pd.read_csv(FILE_PATH)

    def __len__(self):
        return len(self.csv)
    
    def tokenize(self, text):
        return text.split()
    
    def __getitem__(self, idx):
        text = self.csv.iloc[idx, :]
        Image_Name = text.Image_Name
        Recipe_title = text.Title
        ingredients = text.Cleaned_Ingredients
        tokens = self.tokenize(Recipe_title)
        return tokens, Image_Name
    
def collate_fn(batch):
    tokens, Image_Name = zip(*batch)
    return tokens, Image_Name

def main(batch_size=3):
    img_set = ImageDataset()
    img_load = torch.utils.data.DataLoader(img_set, batch_size=batch_size, shuffle=True)
    for img in img_load:
        plt.imshow(img[0].permute(1,2,0))
        plt.show()
        break
        
    contrast_data = ContrastiveImageDataset()
    contrast_load = torch.utils.data.DataLoader(contrast_data, batch_size=batch_size, shuffle=True)
    fig, ax = plt.subplots(1,2, figsize=(14, 5))
    for img, contrast_img in contrast_load:
        ax[0].imshow(denormalize(img[0].permute(1,2,0)))
        ax[1].imshow(denormalize(contrast_img[0].permute(1,2,0)))
        plt.show()
        break

    data_loader = TextDataSet()
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    for tokens, Image_Name in data_loader:
        print(tokens)
        print(Image_Name)
        break

if __name__ == '__main__':
    main()