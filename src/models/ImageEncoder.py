#############################
#### NOTES FOR THE GROUP ####
#############################

# Possible encoders: ViT*, ResNet(50)*, VGG16, AlexNet, EfficientNet*, etc. 
# * = Recommended in the project description

# images are 274 x 169

import torch
import torch._utils
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
from vit_blank import *


def get_image_encoder(args, device_:torch.device) -> nn.Module:

    class ResNet50(nn.Module):

        def __init__(self, output_dim:int, device:torch.device):
            """ Image encoder to obtain features from images. Contains pretrained Resnet50 with last layer removed 
                and a linear layer with the output dimension of (BATCH, image_emb_dim)

            Args:
                image_emb_dim (int): final output dimension of features
                
                device (torch.device)
            """
            
            super(ResNet50, self).__init__()
            self.output_dim = output_dim
            self.device = device
            
            print(f"Encoder:\n \
                    Encoder dimension: {self.output_dim}")
            
            # pretrained Resnet50 model with freezed parameters
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            for param in resnet.parameters(): 
                param.requires_grad_(False)
            
            # remove last layer 
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)
            
            # define an embedding size to map to
            self.fc = nn.Linear(resnet.fc.in_features, self.output_dim) 
            
            
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            """ Forward operation of encoder, passing images through resnet and then linear layer.

            Args:
                > images (torch.Tensor): (BATCH, 3, 224, 224)

            Returns:
                > features (torch.Tensor): (BATCH, IMAGE_EMB_DIM)
            """
            
            features = self.resnet(images)
            #print(features.shape)
            # features: (BATCH, 2048, 1, 1)
            
            features = features.reshape(features.size(0), -1).to(self.device)
            #print(f"Reshaped: {features.shape}")
            # features: (BATCH, 2048)
            
            features = self.fc(features).to(self.device)
            # features: (BATCH, IMAGE_EMB_DIM)
            
            return features
        
    class ViT_Base(nn.Module):

        def __init__(self, output_dim:int, device:torch.device):
            """ Image encoder to obtain features from images. Contains pretrained Resnet50 with last layer removed 
                and a linear layer with the output dimension of (BATCH, image_emb_dim)

            Args:
                image_emb_dim (int): final output dimension of features
                
                device (torch.device)
            """
            
            super(ViT_Base, self).__init__()
            self.output_dim = output_dim
            self.device = device
            
            print(f"Encoder:\n \
                    Encoder dimension: {self.output_dim}")
            
            # pretrained Resnet50 model with freezed parameters
            self.vit = models.vit_b_16(weights='DEFAULT')
            for param in self.vit.parameters(): 
                param.requires_grad_(False)
                
            # redefine heads layer
            self.vit.heads = nn.Sequential(nn.Linear(in_features=self.vit.heads.head.in_features,
                                            out_features=self.output_dim))

            
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            """ Forward operation of encoder, passing images through resnet and then linear layer.

            Args:
                > images (torch.Tensor): (BATCH, 3, 224, 224)

            Returns:
                > features (torch.Tensor): (BATCH, IMAGE_EMB_DIM)
            """
            
            features = self.vit(images) # [1, 512]
            
    class EfficientNet(nn.Module):
        
        def __init__(self, output_dim:int, device:torch.device):
            """ Image encoder to obtain features from images. Contains pretrained Resnet50 with last layer removed 
                and a linear layer with the output dimension of (BATCH, image_emb_dim)

            Args:
                image_emb_dim (int): final output dimension of features
                
                device (torch.device)
            """
            
            super(EfficientNet, self).__init__()
            self.output_dim = output_dim
            self.device = device
            
            print(f"Encoder:\n \
                    Encoder dimension: {self.output_dim}")
            
            # pretrained Resnet50 model with freezed parameters
            self.en = models.efficientnet_b0(weights='DEFAULT')
            for param in self.en.parameters(): 
                param.requires_grad_(False)
                
            # redefine heads layer
            self.en.classifier = nn.Sequential(
                                                nn.Dropout(p=0.2, inplace=True),
                                                nn.Linear(  
                                                            in_features=self.en.classifier[1].in_features,
                                                            out_features=self.output_dim,
                                                        ),
                                               )

            
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            """ Forward operation of encoder, passing images through resnet and then linear layer.

            Args:
                > images (torch.Tensor): (BATCH, 3, 224, 224)

            Returns:
                > features (torch.Tensor): (BATCH, IMAGE_EMB_DIM)
            """
            
            features = self.en(images) # [1, 512]
            #print(features.shape)

            return features
    
    
    if args.img_encoder_name == 'resnet':
        return ResNet50(output_dim=args.embeding_dim, device=device_)
    elif args.img_encoder_name == 'vit':
        return ViT_Base(output_dim=args.embeding_dim, device=device_)
    elif args.img_encoder_name == 'efficientnet':
        return EfficientNet(output_dim=args.embeding_dim, device=device_)
    elif args.img_encoder_name == 'vit_blank':
        config = {
            "patch_size": 16,  # Input image size: 224x224 -> 14x14 patches
            "hidden_size": 48,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4 * 48, # 4 * hidden_size
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "initializer_range": 0.02,
            "image_size": 224,
            "num_classes": 512, # num_classes of CIFAR10
            "num_channels": 3,
            "qkv_bias": True,
            "use_faster_attention": True,
        }
        # These are not hard constraints, but are used to prevent misconfigurations
        assert config["hidden_size"] % config["num_attention_heads"] == 0
        assert config['intermediate_size'] == 4 * config['hidden_size']
        assert config['image_size'] % config['patch_size'] == 0   
        return ViTForClassfication(config)
    else:
        print("Failure to procure model! Please use the following options:\n'resnet', 'vit', 'efficientnet' or 'vit_blank'.")
    
        




"""

###################
# Jakob stuff #

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
        self.path = 'C:/Users/jakob/Desktop/UniStuff/ADLCV-recipe-retrieval/src/data/processed/Food Images'
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
    '''returns: Image_Name, title, ingredients, instructions'''
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
        instructions = text.Instructions
        title, ingredients, instructions = self.tokenize(Recipe_title), self.tokenize(ingredients), self.tokenize(instructions)
        return Image_Name, title, ingredients, instructions
    
def collate_fn(batch):
    Image_Name, title, ingredients, instructions = zip(*batch)
    return Image_Name, title, ingredients, instructions


    
    
img_set = ImageDataset()
img_load = torch.utils.data.DataLoader(img_set, batch_size=3, shuffle=True)
for img in img_load:
    plt.imshow(denormalize(img[0].permute(1,2,0)))
    plt.show()
    break



## Testing stuff

device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_image_encoder(choice='efficientnet')

for img in img_load:
    x = img[0]
    break


"""