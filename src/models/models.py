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
from src.utils import get_loss_fn, denormalise_batch
from src.data.dataloader import denormalize

from src.models.heads import get_head
import clip  
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import textwrap
from torchvision.transforms.functional import to_pil_image


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

        if self.args.lr_scheduler:
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

        # ------------ Plots of OOD data ----------------

        # Testing Smoerrebroed
        if self.args.ood_phillip_data:
            phi_R = torch.cat(torch.load('/Users/philliphoejbjerg/Desktop/models/test_embeddings.pt'))
            # Loading and flattening recipes
            R = [item for sublist in torch.load('/Users/philliphoejbjerg/Desktop/models/test_strings.pt') for item in sublist]

            if self.args.CLIP:
                smoerrebroeds_recipe = 'Smørrebrød (Danish Open-faced Sandwiches) Seafood'
            else:
                smoerrebroeds_recipe = '''Smørrebrød (Danish Open-faced Sandwiches) Seafood, 1 jar marinated herring (you can get this at IKEA or possibly at your local grocery store!), 1 pack frozen baby shrimp (you can get this at IKEA or possibly at your local grocery store - you are looking for small shrimp!), 1 pack smoked salmon (usually come in 4oz packs) (get your favorite kind!), 1-2 tins mackerel in tomato sauce (you can get these from Scandinavian stores, or in a pinch, you can find tinned mackerel in oil in most grocery stores!), Meat & Eggs, 1 leverpostej/liver pate (we have a recipe if you would like to make a homemade version, or you can find liver pate in most grocery stores!), Frikadeller/Danish meatballs (we have a recipe for homemade danish meatballs here; the recipe will yield leftover meatballs!), 1 pack roast beef (we just get pre-packaged deli meat, ~8oz!), 4 hardboiled eggs, Vegetables, 8 small boiled potatoes, 4 campari sized tomatoes, 1 cucumber, 1 avocado, 1 head Butter lettuce (any crunchy, larger leaf lettuce will work!), 4-5 radishes (this is optional but we love the crunch the radishes add!), 1-2 pickles (optional), Toppings, Condiments, and Garnishes, 1 jar pickled beets (you can find pickled beets at Scandinavian stores, or see our recipe here!), Capers (optional), Mayonnaise (optional), Danish Remoulade (you can find remoulade at Scandinavian stores, or see our recipe here!), Butter (for spreading on the bread), Sour Cream (optional), Fried Onions (you can find these at IKEA, Trader Joes, or other grocery stores!), Fresh dill, Chives, Micro greens (optional, but they add great texture and flavor!), Sliced red onion (even better if you have time to pickle the red onions; you can also rinse them to remove some of the sharp onion flavor!), 1 lemon, Bread, 2 loaves of danish rye bread (we have a recipe here for rugbrød; this may seem like a lot, but you can always eat leftover bread! you can also often find pre-sliced seeded rye bread in grocery stores, or premade mixes at Scandinavian stores!), Prepare for your meal by making ahead certain items, either the day before your meal, or the morning of. These items (if making them from scratch) include liver pate, danish meatballs, remoulade, pickled beets, and the rye bread. Slice and prepare all of the vegetables, wash the lettuce and herbs, and prepare the garnishes by either setting them on the table or placing in serving containers/bowls. Boil potatoes and eggs as well. Slice the rye bread a little over 1/4" thick. Now you can either arrange all the ingredients including the fish and the meats on serving platters and allow guests to build their own sandwiches, or you can prepare each version of the sandwich for everyone at a time. We like to put everything out and prepare our own sandwiches starting with fish, then meats, and then vegetables. Refer to our list above to see how we like to construct each sandwich! And if you're like us, buckle up for a lunch that probably lasts six hours!'''
            R.append(smoerrebroeds_recipe)

            import torchvision.transforms as T
            from PIL import Image
            import glob

            transform = T.Compose([
                T.Resize((274, 169)),
                T.CenterCrop(224),
                T.RandomRotation((270,270)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])        

            image_list = []
            for filename in glob.glob('/Users/philliphoejbjerg/Desktop/food/*.jpg'): #assuming jpeg
                im=transform(Image.open(filename).convert("RGB"))
                image_list.append(im)

            # Getting latent space representations
            OOD_z = self.img_encoder(torch.stack(image_list))
            smoerebroed_z = self.R_encoder(R[-1])
            
            # Normalize pretrained embeddings
            if self.args.normalize:
                OOD_z = nn.functional.normalize(OOD_z, p=2, dim=-1)
                smoerebroed_z = nn.functional.normalize(smoerebroed_z, p=2, dim=-1)                

            # Project to embedding space
            phi_OOD = self.W_img(OOD_z)
            phi_smoerebroed = self.W_R(smoerebroed_z)

            sim = pairwise_cosine_similarity(phi_OOD, torch.cat([phi_R, phi_smoerebroed]))   

            OOD_top_preds = torch.topk(sim, k=len(R), dim=1)[1]

            max_k = 5
            text_width = 25

            fig, ax = plt.subplots(len(image_list), max_k+1, dpi=200, figsize=(15, 7),tight_layout=True)
            for i, top_preds in enumerate(OOD_top_preds[:,:max_k]):

                ax[0].imshow(denormalize(image_list[i].permute(1, 2, 0).cpu()))
                ax[0].axis('off')
                
                for j, pred in enumerate(top_preds):
                    wc = WordCloud(background_color='white',width=274,height=169).generate(R[pred.item()])        
                    ax[j+1].imshow(wc, interpolation='bilinear')    
                    ax[j+1].axis('off')
                    # adding title
                    # ax[i,j+1].set_title(textwrap.fill(self.test_dataloader_.dataset.csv.iloc[pred.item(),:].Title, 8), wrap=True)
                    # ax[i,j+1].set_title(self.test_dataloader_.dataset.csv.iloc[pred.item(),:].Title, wrap=True, fontsize=8)

            os.makedirs(f'reports/figures/{self.args.experiment_name}', exist_ok=True)
            plt.savefig(f'reports/figures/{self.args.experiment_name}/smoerebrød.png', dpi=300, bbox_inches='tight')
    
        # --------------------   ACTUAL PLOTS -----------------------     
        # Calculate cosine similarity
        cosine_similarities = pairwise_cosine_similarity(phi_img, phi_R)

        # first row is the first img wrt all recipes      
        R_pred = torch.argmax(cosine_similarities, dim = 1)

        # first column is the first recipe wrt all images 
        img_pred  = torch.argmax(cosine_similarities, dim = 0) 

        if batch_idx == 0:    
            # tensorboard embedding projector  
            title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size:batch_idx*batch_size+batch_size,:].Title
            label_img = denormalise_batch(img)  # the original images
            self.logger.experiment.add_embedding(phi_img, metadata=title, label_img=label_img)

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

class CLIP(L.LightningModule):    
    def __init__(self, 
                 train_dataloader, 
                 val_dataloader, 
                 test_dataloader,
                 batch_size = 64,
                 # embedding_dim = 256,
                 args = None):
        
        super().__init__()
        self.save_hyperparameters()
        self.device_            = 'cuda' if torch.cuda.is_available() else 'cpu'

        from transformers import CLIPProcessor, CLIPModel, CLIPConfig, CLIPTextConfig, CLIPVisionConfig      

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device_)

        self.train_dataloader_  = train_dataloader
        self.val_dataloader_    = val_dataloader
        self.test_dataloader_   = test_dataloader
        self.args               = args
        
        # Defining accuracy metric - depends on size of testset batch
        self.accuracy = Accuracy(task="multiclass", num_classes=self.test_dataloader_.batch_size)  

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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

        image = torch.stack([self.preprocess(to_pil_image(image)).to(self.device_) for image in img])
        text = clip.tokenize(R).to(self.device_)  

        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)

        return image_features, text_features

    # Training step
    def training_step(self, batch, batch_idx):
        # Not needed for pretrained clip
        return 0
    
    def validation_step(self, batch, batch_idx):
        # Not needed for pretrained clip
        return 0

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
        metrics['R_med_r'] = np.median(sorted(R_positions + 1))
        metrics['img_med_r'] = np.median(sorted(img_positions + 1))

        self.log_dict(metrics, batch_size=batch_size)

    def predict_step(self, batch, batch_idx):

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

                if not os.path.isdir(f'reports/figures/{self.args.experiment_name}/'):
                    os.makedirs(f'reports/figures/{self.args.experiment_name}/')
                plt.savefig(f'reports/figures/{self.args.experiment_name}/model_examples_{n_images/4}.png', dpi=300, bbox_inches='tight')
                # plt.show()
                plt.close('all')

        return R_pred, img_pred   