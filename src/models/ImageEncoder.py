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

        def __init__(self, args, device:torch.device):
            """ Image encoder to obtain features from images. Contains pretrained Resnet50 with last layer removed 
                and a linear layer with the output dimension of (BATCH, image_emb_dim)

            Args:
                image_emb_dim (int): final output dimension of features
                
                device (torch.device)
            """
            
            super(ResNet50, self).__init__()
            self.device = device
            
            # pretrained Resnet50 model with freezed parameters
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            if args.freeze_models:
                for param in resnet.parameters(): 
                    param.requires_grad_(False)
            
            # remove last layer 
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)

            self.output_dim = resnet.fc.in_features

            print(f"Encoder:\n \
                    Encoder dimension: {self.output_dim}")            

            """
            # define an embedding size to map to
            self.fc = nn.Linear(resnet.fc.in_features, self.output_dim) 
            if self.full_freeze:
                for param in self.fc.parameters():
                    param.requires_grad_(False)
            """
            
            
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
            
            #features = self.fc(features).to(self.device)
            # features: (BATCH, IMAGE_EMB_DIM)
            
            return features
        
    class ViT_Base(nn.Module):

        def __init__(self, args, device:torch.device):
            """ Image encoder to obtain features from images. Contains pretrained Resnet50 with last layer removed 
                and a linear layer with the output dimension of (BATCH, image_emb_dim)

            Args:
                image_emb_dim (int): final output dimension of features
                
                device (torch.device)
            """
            
            super(ViT_Base, self).__init__()
            self.device = device
            
            # pretrained Resnet50 model with freezed parameters
            self.vit = models.vit_b_16(weights='DEFAULT')

            if args.freeze_models:
                for param in self.vit.parameters(): 
                    param.requires_grad_(False)

            # Output dim for the projection head
            self.output_dim = 1000

            print(f"Encoder:\n \
                    Encoder dimension: {self.output_dim}")                        
                
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            """ Forward operation of encoder, passing images through resnet and then linear layer.

            Args:
                > images (torch.Tensor): (BATCH, 3, 224, 224)

            Returns:
                > features (torch.Tensor): (BATCH, IMAGE_EMB_DIM)
            """
            
            features = self.vit(images)
            return features
            
    class EfficientNet(nn.Module):
        
        def __init__(self, device:torch.device):
            """ Image encoder to obtain features from images. Contains pretrained Resnet50 with last layer removed 
                and a linear layer with the output dimension of (BATCH, image_emb_dim)

            Args:
                image_emb_dim (int): final output dimension of features
                
                device (torch.device)
            """

            super(EfficientNet, self).__init__()
            self.device = device            
            
            # Following is a fix to error: https://github.com/pytorch/vision/issues/7744
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            from torchvision.models._api import WeightsEnum
            from torch.hub import load_state_dict_from_url

            def get_state_dict(self, *args, **kwargs):
                kwargs.pop("check_hash")
                return load_state_dict_from_url(self.url, *args, **kwargs)
            WeightsEnum.get_state_dict = get_state_dict
            
            # pretrained Resnet50 model with freezed parameters
            self.en = efficientnet_b0(weights="DEFAULT")
            for param in self.en.parameters(): 
                param.requires_grad_(False)

            self.output_dim = 1000

            print(f"Encoder:\n \
                    Encoder dimension: {self.output_dim}")            
                
            # redefine heads layer
            """
            self.en.classifier = nn.Sequential(
                                                nn.Dropout(p=0.2, inplace=True),
                                                nn.Linear(  
                                                            in_features=self.en.classifier[1].in_features,
                                                            out_features=self.output_dim,
                                                        ),
                                               )
            """
            
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
        return ResNet50(args, device=device_)
    elif args.img_encoder_name == 'vit':
        return ViT_Base(args, device=device_)
    elif args.img_encoder_name == 'efficientnet':
        return EfficientNet(args, device=device_)
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