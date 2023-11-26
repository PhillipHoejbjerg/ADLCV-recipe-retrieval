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

torch.set_float32_matmul_precision('high')

class RecipeRetrievalLightningModule(L.LightningModule):
    def __init__(self, 
                 img_encoder, 
                 R_encoder, 
                 projection_head,
                 train_dataloader, 
                 val_dataloader, 
                 test_dataloader,
                 loss_fn, 
                 lr = 0.001,
                 batch_size = 64,
                 embedding_dim = 256,
                 args = None):
        
        super().__init__()
        self.save_hyperparameters('lr', 'batch_size', 'embedding_dim', 'loss_fn')

        self.loss_function      = loss_fn
        self.img_encoder        = img_encoder
        self.R_encoder          = R_encoder
        self.train_dataloader_  = train_dataloader
        self.val_dataloader_    = val_dataloader
        self.test_dataloader_   = test_dataloader
        self.device_            = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args               = args

        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # Projection layers - mapping to embedding space
        self.W_R   = projection_head(R_encoder.output_dim,   self.embedding_dim)
        self.W_img = projection_head(img_encoder.output_dim, self.embedding_dim)  
        
        # Learnable temperature parameter for ClipLoss
        if args.loss_fn == 'ClipLoss':
            self.t     = torch.tensor([args.temperature],device=self.device_) if args.temperature else torch.nn.Parameter(torch.tensor([1.0],device=self.device_))

        # Defining accuracy metric - depends on size of testset batch
        self.accuracy = Accuracy(task="multiclass", num_classes=self.test_dataloader_.batch_size) 

    
    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Define the learning rate scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True),
            'monitor': 'val_loss',  # Replace with the metric you want to monitor
            'interval': 'epoch',
            'frequency': 1
        }

        if args.lr_scheduler:
            return [optimizer], [scheduler] 
        else:
            return optimizer

    # Dataloaders
    def train_dataloader(self):
        return self.train_dataloader_
    def val_dataloader(self):
        return self.val_dataloader_ 
    def predict_dataloader(self):
        return self.test_dataloader_
    def test_dataloader(self):
        return self.test_dataloader_ 

    # Forward pass
    def forward(self, img, R):
        
        # Getting latent space representations
        img_z, R_z = self.img_encoder(img), self.R_encoder(R)
        
        # Normalize pretrained embeddings
        if self.args.normalize:
            img_z, R_z = nn.functional.normalize(img_z, p=2, dim=-1), nn.functional.normalize(R_z, p=2, dim=-1)

        # Project to embedding space
        phi_img, phi_R = self.W_img(img_z), self.W_R(R_z)
        
        return phi_img, phi_R

    # Training step
    def training_step(self, batch, batch_idx):
        
        # Unpacking batch
        img, R, is_pos_pair = batch

        # Getting latent space representations
        phi_img, phi_R = self(img, R)

        # Calculate loss - inputs depend on loss function
        if self.loss_function.__class__.__name__ == 'ClipLoss':
            loss = self.loss_function(phi_img, phi_R, self.t)
        elif self.loss_function.__class__.__name__ == 'TripletLoss':
            loss = self.loss_function(phi_img, phi_R)
        else:
            loss = self.loss_function(phi_img, phi_R, torch.where(is_pos_pair, torch.tensor(1), torch.tensor(-1)))

        # Logging loss
        self.log('train_loss', loss, batch_size=img.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        # Unpacking batch
        img, R, is_pos_pair = batch

        # Getting latent space representations
        phi_img, phi_R = self(img, R)

        # Calculate loss - inputs depend on loss function
        if self.loss_function.__class__.__name__ == 'ClipLoss':
            loss = self.loss_function(phi_img, phi_R, self.t)
        elif self.loss_function.__class__.__name__ == 'TripletLoss':
            loss = self.loss_function(phi_img, phi_R)
        else:
            loss = self.loss_function(phi_img, phi_R, torch.where(is_pos_pair, torch.tensor(1), torch.tensor(-1)))

        # Logging loss
        self.log('val_loss', loss, batch_size=img.shape[0])

        return loss

    def test_step(self, batch, batch_idx, recall_klist=(1, 5, 10, 100, 250)):
        assert len(recall_klist) > 0, "recall_klist cannot be empty"
        metrics = {}

        # largest k to compute recall
        max_k = int(max(recall_klist))

        # Unpacking batch
        img, R, _ = batch 
        batch_size = img.shape[0]

        # Mapping to embedding space
        phi_img, phi_R = self(img, R)

        # --------------------        
        # Calculate cosine similarity
        cosine_similarities = pairwise_cosine_similarity(phi_img, phi_R)

        # first row is the first img wrt all recipes      
        R_pred = torch.argmax(cosine_similarities, dim = 1)

        # first column is the first recipe wrt all images 
        img_pred  = torch.argmax(cosine_similarities, dim = 0)

        # Calculating accuracy
        R_acc   = self.accuracy(R_pred,   torch.arange(batch_size).to(self.device_))
        img_acc = self.accuracy(img_pred, torch.arange(batch_size).to(self.device_))
        metrics['R_acc'] = R_acc
        metrics['img_acc'] = img_acc

        # --------------------

        # Calculating recall @ k
        R_top_preds = torch.topk(cosine_similarities, k=batch_size, dim=1)[1] #[:,:k]
        img_top_preds = torch.topk(cosine_similarities, k=batch_size, dim=0)[1].T #[:k,:]

        # positions, i.e. the index of the positive element in the topk
        R_positions = torch.Tensor([(i == R_top_preds[i]).nonzero().squeeze(0) for i in torch.arange(batch_size)])
        img_positions = torch.Tensor([(i == img_top_preds[i]).nonzero().squeeze(0) for i in torch.arange(batch_size)])

        # Recall @ k
        for k in recall_klist:
            metrics[f'R_recall_{int(k)}'] = np.mean((R_positions < k).cpu().numpy())
            metrics[f'img_recall_{int(k)}'] = np.mean((img_positions < k).cpu().numpy())

        # median ranking:
        metrics['R_med_r'] = np.median(sorted(R_positions))
        metrics['img_med_r'] = np.median(sorted(img_positions))

        self.log_dict(metrics, batch_size=batch_size)

    def predict_step(self, batch, batch_idx):

        # Unpacking batch
        img, R, _ = batch 
        batch_size = img.shape[0]

        # Mapping to embedding space
        phi_img, phi_R = self(img, R)
        def denormalise_batch(imgs):
            '''denormalise a batch of images based on imagenet stats'''
            imgs = imgs.clone()
            imgs[:, 0, :, :] = imgs[:, 0, :, :] * 0.229 + 0.485
            imgs[:, 1, :, :] = imgs[:, 1, :, :] * 0.224 + 0.456
            imgs[:, 2, :, :] = imgs[:, 2, :, :] * 0.225 + 0.406
            return imgs

        features = phi_img  # the embeddings
        title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size:batch_idx*batch_size+batch_size,:].Title
        labels = title # the titles
        label_img = denormalise_batch(img)  # the original images
        self.logger.experiment.add_embedding(features, metadata=labels, label_img=label_img)

        # --------------------        
        # Calculate cosine similarity
        cosine_similarities = pairwise_cosine_similarity(phi_img, phi_R)

        # first row is the first img wrt all recipes      
        R_pred = torch.argmax(cosine_similarities, dim = 1)

        # first column is the first recipe wrt all images 
        img_pred  = torch.argmax(cosine_similarities, dim = 0)  

        if batch_idx == 0:      

            max_k = 5

            R_top_preds = torch.topk(cosine_similarities, k=batch_size, dim=1)[1] #[:,:k]
            img_top_preds = torch.topk(cosine_similarities, k=batch_size, dim=0)[1].T #[:k,:]

            # positions, i.e. the index of the positive element in the topk
            R_positions = torch.Tensor([(i == R_top_preds[i]).nonzero().squeeze(0) for i in torch.arange(batch_size)])
            img_positions = torch.Tensor([(i == img_top_preds[i]).nonzero().squeeze(0) for i in torch.arange(batch_size)])

            text_width = 25
            # image and wordclouds  
            # Plot the closest text as a wordcloud
            for n_images in range(0, 40, 4):
                fig, ax = plt.subplots(8, max_k+1, dpi=200, figsize=(15, 7),tight_layout=True)
                for j in range(4):
                    rdn_img_to_plot = n_images+j

                    ax[2*j,0].imshow(denormalize(img[rdn_img_to_plot].permute(1, 2, 0).cpu()))
                    ax[2*j,0].axis('off')
                    # ax[2*j,0].set_title(textwrap.fill(self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+rdn_img_to_plot,:].Title, text_width), wrap=True)
                    rect = plt.Rectangle((ax[j,0].get_xlim()[0], ax[j,0].get_ylim()[0]), ax[j,0].get_xlim()[1]-ax[j,0].get_xlim()[0], ax[j,0].get_ylim()[1]-ax[j,0].get_ylim()[0],linewidth=5,edgecolor='b',facecolor='none')
                    ax[2*j,0].add_patch(rect)  

                    for i in range(max_k):
                        recipe_no = R_top_preds[rdn_img_to_plot][i].item()
                        closest_text = R[recipe_no]
                        closest_text_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+recipe_no,:].Title
                        # 274x169
                        wordcloud = WordCloud(background_color='white',width=274,height=169).generate(closest_text)

                        ax[2*j,i+1].imshow(wordcloud, interpolation='bilinear')
                        # ax[2*j,i+1].set_title(textwrap.fill(closest_text_title, text_width), wrap=True)
                        ax[2*j,i+1].axis('off')
                        # make rectangle around this ax
                        if R_positions[rdn_img_to_plot].item() == i:
                            # Draw rectangle around subbplot
                            rect = plt.Rectangle((ax[j,i+1].get_xlim()[0], ax[j,i+1].get_ylim()[0]), ax[j,i+1].get_xlim()[1]-ax[j,i+1].get_xlim()[0], ax[j,i+1].get_ylim()[1]-ax[j,i+1].get_ylim()[0],linewidth=5,edgecolor='g',facecolor='none')
                            ax[2*j,i+1].add_patch(rect)

                    text = R[rdn_img_to_plot]
                    wordcloud = WordCloud(background_color='white',width=274,height=169).generate(text)
                    closest_text_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+rdn_img_to_plot,:].Title

                    ax[2*j+1,0].imshow(wordcloud, interpolation='bilinear')
                    ax[2*j+1,0].axis('off')
                    # ax[2*j+1,0].set_title(textwrap.fill(closest_text_title, text_width), wrap=True)
                    rect = plt.Rectangle((ax[j+1,0].get_xlim()[0], ax[j+1,0].get_ylim()[0]), ax[j+1,0].get_xlim()[1]-ax[j+1,0].get_xlim()[0], ax[j+1,0].get_ylim()[1]-ax[j+1,0].get_ylim()[0],linewidth=5,edgecolor='b',facecolor='none')
                    ax[2*j+1,0].add_patch(rect)                
                    
                    for i in range(max_k):
                        img_no = img_top_preds[rdn_img_to_plot][i].item()
                        closest_image = img[img_no]
                        closest_image_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+img_no,:].Title
                        ax[2*j+1,i+1].imshow(denormalize(closest_image.permute(1, 2, 0).cpu()))
                        ax[2*j+1,i+1].axis('off')
                        #ax[2*j+1,i+1].set_title(textwrap.fill(closest_image_title, text_width), wrap=True)
                        # make rectangle around this ax
                        if img_positions[rdn_img_to_plot].item() == i:
                            # Draw rectangle around subbplot
                            rect = plt.Rectangle((ax[j+1,i+1].get_xlim()[0], ax[j+1,i+1].get_ylim()[0]), ax[j+1,i+1].get_xlim()[1]-ax[j+1,i+1].get_xlim()[0], ax[j+1,i+1].get_ylim()[1]-ax[j+1,i+1].get_ylim()[0],linewidth=5,edgecolor='g',facecolor='none')
                            ax[2*j+1,i+1].add_patch(rect)
                os.makedirs(f'reports/figures/{self.args.experiment_name}', exist_ok=True)
                plt.savefig(f'reports/figures/{self.args.experiment_name}/model_examples_{n_images/4}.png', dpi=300, bbox_inches='tight')
                # plt.show()
                plt.close('all')

        return R_pred, img_pred

if __name__ == "__main__":
       
    parser = argparse.ArgumentParser(description='Recipe Retrieval Training Script')
    
    # Experiment name
    parser.add_argument('--experiment_name', type=str, default="test", help='Experiment name - default test')
    
    # Experiment modes
    parser.add_argument("--text_mode", type=str, default='title,ingredients,instructions', help="text mode - default title")

    # Encoders
    parser.add_argument('--img_encoder_name', type=str, default="resnet", help='resnet, vit, efficientnet')
    parser.add_argument('--text_encoder_name', type=str, default="roberta_base", help='roberta_base, transformer_base')
    parser.add_argument('--head_name', type=str, default="linear", help='projection_head')

    # Encoder Settings
    parser.add_argument('--embedding_dim', type=int, default=256, help='embedding dim - default 256')
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=True) # --normalize or --no-normalize
    parser.add_argument('--center_crop', action=argparse.BooleanOptionalAction, default=True) # --center_crop or --no-center_crop
    
    # Loss
    parser.add_argument('--loss_fn', type=str, default="ClipLoss", help='Loss_fn - default cosine')
    parser.add_argument('--margin', type=float, default=0.3, help='margin (for loss function) - default 0.3')
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help='https://velog.io/@clayryu328/paper-review-CLIP-Learning-Transferable-Visual-Models-From-Natural-Language-Supervision')

    # Training params
    parser.add_argument('--p', type=float, default=0.0, help='probability of negative pair - default 0.8')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate - default 0.001')
    parser.add_argument('--batch_size',    type=int, default=401, help='batch size - default 64')
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

    # Get encoders
    img_encoder      = get_image_encoder(args, device)
    R_encoder        = get_text_encoder(args, device) 
    projection_head  = get_head(args, device)

    # Get dataloaders
    train_dataloader = get_dataloader(args, mode = 'train')
    val_dataloader   = get_dataloader(args, mode = 'val')
    test_dataloader  = get_dataloader(args, mode = 'test')    

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

    # # Fitting model
    # trainer.fit(model = model)

    # # Testing model
    # trainer.test(model = model)
    trainer.predict(model = model, ckpt_path='tensorboard_logs/poster/version_4/checkpoints/epoch=17-step=21204.ckpt')
    # trainer.predict(model = model, ckpt_path='tensorboard_logs/poster/version_3/checkpoints/epoch=41-step=49476.ckpt')

# python src/models/train_model.py     --experiment_name triplet_proj_100_testing    --num_epochs 50     --loss_fn TripletLoss     --num_workers 0     --head_name linear     --text_mode "title,ingredients"     --lr_scheduler     --normalize     --center_crop    --batch_size 64