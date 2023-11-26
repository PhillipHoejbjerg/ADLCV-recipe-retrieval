import torch
import numpy as np
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from lightning.pytorch.callbacks import RichProgressBar
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.data.dataloader import get_dataloader
from src.models.ImageEncoder import get_image_encoder
from src.models.text_encoder import get_text_encoder
from src.utils import get_loss_fn
from src.data.dataloader import denormalize

from src.models.heads import get_head
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import textwrap
from torchvision.transforms.functional import to_pil_image

from src.models.models import CLIP, RecipeRetrievalLightningModule

torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
       
    parser = argparse.ArgumentParser(description='Recipe Retrieval Training Script')

    # Baselining with pre-trained CLIP?
    parser.add_argument('--CLIP', action=argparse.BooleanOptionalAction, default=False)
    
    # Experiment name
    parser.add_argument('--experiment_name', type=str, default="test", help='Experiment name - default test')
    
    # Experiment modes
    parser.add_argument("--text_mode", type=str, default='title ingredients instructions', help="text mode - default title")

    # Encoders
    parser.add_argument('--img_encoder_name', type=str, default="resnet", help='resnet, vit, efficientnet')
    parser.add_argument('--text_encoder_name', type=str, default="roberta_base", help='roberta_base, transformer_base')
    parser.add_argument('--head_name', type=str, default="projection_head", help='projection_head')

    # Encoder Settings
    parser.add_argument('--embedding_dim', type=int, default=256, help='embedding dim - default 256')
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=True) # --normalize or --no-normalize
    parser.add_argument('--center_crop', action=argparse.BooleanOptionalAction, default=True) # --center_crop or --no-center_crop
    
    # Loss
    parser.add_argument('--loss_fn', type=str, default="ClipLoss", help='Loss_fn - default cosine')
    parser.add_argument('--margin', type=float, default=0.5, help='margin (for loss function) - default 0.5')
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help='https://velog.io/@clayryu328/paper-review-CLIP-Learning-Transferable-Visual-Models-From-Natural-Language-Supervision')

    # Training params
    parser.add_argument('--p', type=float, default=0.0, help='probability of negative pair - default 0.8')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate - default 0.001')
    parser.add_argument('--batch_size',    type=int, default=8, help='batch size - default 64')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs - default 20')
    parser.add_argument('--lr_scheduler', action=argparse.BooleanOptionalAction, default=False) # --lr_scheduler or --no-lr_scheduler
    parser.add_argument('--num_workers', type=int, default=11, help='number of workers - default 0')

    parser.add_argument('--model_path', type=str, default=None, help='path to model - default None')

    args = parser.parse_args()

    args.text_mode = [item for item in args.text_mode.split(',')]

    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initializing tensorboard logger
    tb_logger        = TensorBoardLogger(save_dir = "tensorboard_logs", name=args.experiment_name)     

    # -----------

    # Get dataloaders
    train_dataloader = get_dataloader(args, mode = 'train')
    val_dataloader   = get_dataloader(args, mode = 'val')
    test_dataloader  = get_dataloader(args, mode = 'test')        

    # Get CLIP model
    if args.CLIP:
        model = CLIP(train_dataloader, 
                val_dataloader, 
                test_dataloader,
                batch_size = args.batch_size,
                args = args)

    # Get own model
    else:
        # Get encoders
        img_encoder      = get_image_encoder(args, device)
        R_encoder        = get_text_encoder(args, device) 
        projection_head  = get_head(args, device)

        # Defining loss function
        loss_fn = get_loss_fn(args)

        # Defining model
        model = RecipeRetrievalLightningModule(img_encoder, 
                                                R_encoder, 
                                                projection_head,
                                                train_dataloader, 
                                                val_dataloader, 
                                                test_dataloader,
                                                loss_fn, 
                                                lr = args.lr,
                                                batch_size = args.batch_size,
                                                embedding_dim = args.embedding_dim,
                                                args = args)
        
    # Defining callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    trainer = L.Trainer(default_root_dir="/models", # save_dir
                        callbacks=[checkpoint_callback, RichProgressBar()],
                        logger = tb_logger,
                        max_epochs=args.num_epochs,
                        check_val_every_n_epoch=1,)

    # Fitting model
    if not args.CLIP:
        trainer.fit(model = model)

    # Testing model
    trainer.test(model = model)
    trainer.predict(model = model)