import torch
import numpy as np
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
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
from src.models.text_encoder import get_text_encoder # TODO: Does not exist yet
from src.utils import get_loss_fn
from src.data.dataloader import denormalize

from src.models.heads import get_head
from wordcloud import WordCloud


#import os

#os.chdir('C:/Users/jakob/Desktop/UniStuff/ADLCV-recipe-retrieval/src')

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
        self.save_hyperparameters('lr', 'batch_size', 'embedding_dim')

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

        # Mapping the output of the encoders to the embedding space
        # 512 --> 256
        self.W_R   = projection_head(R_encoder.output_dim,   self.embedding_dim)
        self.W_img = projection_head(img_encoder.output_dim, self.embedding_dim)  
        # Learnable temperature parameter
        self.t     = torch.tensor([args.temperature],device=self.device_) if args.temperature else torch.nn.Parameter(torch.tensor([1.0],device=self.device_))

        # Accuracy
        self.accuracy = Accuracy(task="multiclass", num_classes=self.test_dataloader_.batch_size) 

        # To optimise each encoder separately
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html
        # self.automatic_optimization = False #TODO: IMPLEMENT THIS WHEN EVERYTHING ELSE IS RUNNING
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Define the learning rate scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True),
            'monitor': 'val_loss',  # Replace with the metric you want to monitor
            'interval': 'epoch',
            'frequency': 1
        }
        
        if self.args.lr_scheduler:
            print("using lr_schduler!!!ß")
            return [optimizer], [scheduler]
        else:
            return optimizer

    # def configure_optimizers(self):
    #     '''for manual optimization'''
    #     image_optimizer = torch.optim.Adam(self.img_encoder.parameters(), lr=self.lr)
    #     text_optimizer = torch.optim.Adam(self.R_encoder.parameters(), lr=self.lr)
    #     return image_optimizer, text_optimizer

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_ 
    
    def predict_dataloader(self):
        # this is technically the wrong dloader, but for testing purposes it is fine
        # as long as args.p = 0
        # should be test_dataloader_
        return self.test_dataloader_
    
    # Entire test_set - in order to predict on entirety of test set
    def test_dataloader(self):
        return self.test_dataloader_ 

    def forward(self, img, R):
        # Getting latent space representations
        img_z, R_z = self.img_encoder(img), self.R_encoder(R)
        # img_z, R_z = nn.functional.tanh(img_z), nn.functional.tanh(R_z)
        
        img_z_n, R_z_n = nn.functional.normalize(img_z, p=2, dim=-1), nn.functional.normalize(R_z, p=2, dim=-1)

        # Mapping to embedding space
        phi_img, phi_R = self.W_img(img_z_n), self.W_R(R_z_n)
        
        return phi_img, phi_R

    def training_step(self, batch, batch_idx):
        
        # Unpacking batch
        img, R, is_pos_pair = batch
        # image_optimizer, text_optimizer = self.optimizers()
        # image_optimizer.zero_grad()
        # text_optimizer.zero_grad()

        phi_img, phi_R = self.forward(img, R)

        # Calculate loss here
        if self.loss_function.__class__.__name__ == 'ClipLoss':
            loss = self.loss_function(phi_img, phi_R, self.t)
        elif self.loss_function.__class__.__name__ == 'TripletLoss':
            loss = self.loss_function(phi_img, phi_R)
        else:
            loss = self.loss_function(phi_img, phi_R, torch.where(is_pos_pair, torch.tensor(1), torch.tensor(-1)))
        # loss.backward()
        # image_optimizer.step()
        # text_optimizer.step()
        # sometimes the batch size is inconsistent if it is the last batch
        batch_size = img.shape[0]
        self.log("train_loss", loss, batch_size=batch_size)
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        # Unpacking batch
        img, R, is_pos_pair = batch

        # Getting latent space representations
        phi_img, phi_R = self(img, R)
        # print("is_pos_pair:", is_pos_pair[0], "/nphi_img:/n", phi_img[0][:6], "phi_R:/n", phi_R[0][:6])

        # Calculate loss here
        if self.loss_function.__class__.__name__ == 'ClipLoss':
            loss = self.loss_function(phi_img, phi_R, self.t)
        elif self.loss_function.__class__.__name__ == 'TripletLoss':
            loss = self.loss_function(phi_img, phi_R)
        else:
            loss = self.loss_function(phi_img, phi_R, torch.where(is_pos_pair, torch.tensor(1), torch.tensor(-1)))
        # sometimes the batch size is inconsistent if it is the last batch
        batch_size = img.shape[0]
        self.log('val_loss', loss, batch_size=batch_size)

    def test_step(self, batch, batch_idx, recall_klist=(1, 5, 10, 100, 250)):
        assert len(recall_klist) > 0, "recall_klist cannot be empty"
        metrics = {}
        # largest k to compute recall
        max_k = int(max(recall_klist))

        # Unpacking batch
        img, R, _ = batch # TODO: IMPORTANT test dataloader should not have negative pairs!
        batch_size = img.shape[0]

        # Mapping to embedding space
        phi_img, phi_R = self(img, R)

        # COSINE LOSSES:
        
        # Calculate cosine similarity
        cosine_similarities = pairwise_cosine_similarity(phi_img, phi_R)

        # first row is the first img wrt all recipes      
        # - thus argmax per row is the predicted recipe
        R_pred = torch.argmax(cosine_similarities, dim = 1)

        # first column is the first recipe wrt all images 
        # - thus argmax per column is the predicted image
        img_pred  = torch.argmax(cosine_similarities, dim = 0)

        # Calculating accuracy
        # print(R_pred.shape, batch_size)
        R_acc   = self.accuracy(R_pred,   torch.arange(batch_size).to(self.device_))
        img_acc = self.accuracy(img_pred, torch.arange(batch_size).to(self.device_))
        metrics['R_acc'] = R_acc
        metrics['img_acc'] = img_acc
        
        # Recall @ k + median ranking:
        # https://github.com/amzn/image-to-recipe-transformers/blob/main/src/utils/metrics.py
        
        # find the number of elements in the ranking that have a lower distance
        # than the positive element (whose distance is in the diagonal
        # of the distance matrix) wrt the query. this gives the rank for each
        # query. (+1 for 1-based indexing)
        cosine_similarities = cosine_similarities.cpu().numpy()
        positions = np.count_nonzero(cosine_similarities < np.diag(cosine_similarities)[:, None], axis=-1) + 1

        # get the topk elements for each query (topk elements with lower dist)
        rankings = np.argpartition(cosine_similarities, range(max_k), axis=-1)[:, :max_k]

        # positive positions for each query (inputs are assumed to be aligned)
        positive_idxs = np.array(range(cosine_similarities.shape[0]))
        # matrix containing a cumulative sum of topk matches for each query
        # if cum_matches_topk[q][k] = 1, it means that the positive for query q
        # was already found in position <=k. if not, the value at that position
        # will be 0.
        cum_matches_topk = np.cumsum(rankings == positive_idxs[:, None],
                                    axis=-1)

        # pre-compute all possible recall values up to k
        recall_values = np.mean(cum_matches_topk, axis=0)

        # Logging metrics
        metrics['medr'] = np.median(positions)
        
        for index in recall_klist:
            metrics[f'recall_{int(index)}'] = recall_values[int(index)-1]
        
        self.log_dict(metrics, batch_size=batch_size)

    def predict_step(self, batch, batch_idx):

        # Unpacking batch
        img, R, _ = batch # always positive pairs
        batch_size = img.shape[0]

        # Getting latent space representations
        img_z, R_z = self.img_encoder(img), self.R_encoder(R)
        # Mapping to embedding space
        phi_img, phi_R = self.W_img(img_z), self.W_R(R_z)
        # Calculate cosine similarity
        cosine_similarities = pairwise_cosine_similarity(phi_img, phi_R)
        # first row is the first img wrt all recipes      
        # - thus argmax per row is the predicted recipe
        R_pred = torch.argmax(cosine_similarities, dim = 1)
        # first column is the first recipe wrt all images 
        # - thus argmax per column is the predicted image
        img_pred  = torch.argmax(cosine_similarities, dim = 0)
        cosine_similarities = cosine_similarities.cpu().numpy()
        positions = np.count_nonzero(cosine_similarities < np.diag(cosine_similarities)[:, None], axis=-1) + 1
        max_k = 4
        # get the topk elements for each query (topk elements with lower dist)
        rankings_R = np.argpartition(cosine_similarities, max_k, axis=-1)[:, :max_k]
        # torch.topk(torch.tensor(cosine_similarities),max_k,largest=False,sorted=True,dim=-1)
        rankings_img = np.argpartition(cosine_similarities, max_k, axis=0)[:max_k, :].transpose(1,0)
        # with torch (it does the same)
        # torch.topk(torch.tensor(cosine_similarities),max_k,largest=False,sorted=True,dim=0).values.permute(1,0)
        import matplotlib.pyplot as plt
        import textwrap
        text_width = 25
        # image and wordclouds  
        # Plot the closest text as a wordcloud
        for n_images in range(10):
            fig, ax = plt.subplots(2, max_k+1, dpi=200, figsize=(15, 5),tight_layout=True)
            for j in range(2):
                rdn_img_to_plot = n_images+j

                ax[j,0].imshow(denormalize(img[rdn_img_to_plot].permute(1, 2, 0).cpu()))
                ax[j,0].axis('off')
                ax[j,0].set_title(textwrap.fill(self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+rdn_img_to_plot,:].Title, text_width), wrap=True)
                for i in range(max_k):
                    recipe_no = rankings_R[rdn_img_to_plot][i].item()
                    closest_text = R[recipe_no]
                    closest_text_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+recipe_no,:].Title
                    # 274x169
                    wordcloud = WordCloud(background_color='white',width=274,height=169).generate(closest_text)

                    ax[j,i+1].imshow(wordcloud, interpolation='bilinear')
                    ax[j,i+1].set_title(textwrap.fill(closest_text_title, text_width), wrap=True)
                    ax[j,i+1].axis('off')
            plt.savefig(f'reports/figures/im2text_{batch_idx}.png', dpi=300, bbox_inches='tight')
            # plt.show()
            # plot the closest images
            fig, ax = plt.subplots(2, max_k+1, dpi=200, figsize=(15, 5),tight_layout=True)
            for j in range(2):
                rdn_img_to_plot = n_images+j
                text = R[rdn_img_to_plot]
                wordcloud = WordCloud(background_color='white',width=274,height=169).generate(text)
                closest_text_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+rdn_img_to_plot,:].Title

                ax[j,0].imshow(wordcloud, interpolation='bilinear')
                ax[j,0].axis('off')
                ax[j,0].set_title(textwrap.fill(closest_text_title, text_width), wrap=True)
                for i in range(max_k):
                    img_no = rankings_img[rdn_img_to_plot][i].item()
                    closest_image = img[img_no]
                    closest_image_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+img_no,:].Title
                    ax[j,i+1].imshow(denormalize(closest_image.permute(1, 2, 0).cpu()))
                    ax[j,i+1].axis('off')
                    ax[j,i+1].set_title(textwrap.fill(closest_image_title, text_width), wrap=True)
            plt.savefig(f'reports/figures/text2im_{batch_idx}.png', dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close('all')

        return R_pred, img_pred

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recipe Retrieval Training Script')
    
    parser.add_argument('--batch_size',    type=int, default=8, help='batch size - default 64')
    parser.add_argument('--embedding_dim', type=int, default=256, help='embedding dim - default 256')
    parser.add_argument('--full_freeze', type=bool, default=False, help='ie. head of encoder also frozen - default False')
    parser.add_argument('--margin', type=float, default=0.5, help='margin (for loss function) - default 0.5')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate - default 0.001')
    parser.add_argument('--experiment_name', type=str, default="test", help='Experiment name - default test')
    parser.add_argument('--img_encoder_name', type=str, default="resnet", help='resnet, vit, efficientnet')
    parser.add_argument('--text_encoder_name', type=str, default="roberta_base", help='roberta_base, transformer_base')
    parser.add_argument('--head_name', type=str, default="projection_head", help='projection_head')
    parser.add_argument('--loss_fn', type=str, default="ClipLoss", help='Loss_fn - default cosine')
    parser.add_argument('--p', type=float, default=0.0, help='probability of negative pair - default 0.8')
    parser.add_argument("--text_mode", type=str, default='title ingredients instructions', help="text mode - default title")
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads - default 4')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs - default 20')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers - default 4')
    parser.add_argument('--pos_enc', type=str, default='fixed', help='positional encoding - default fixed')
    parser.add_argument('--pool', type=str, default='max', help='pooling - default max')
    parser.add_argument('--dropout', type=float, default=0.0, help='probability of dropout - default 0.0')
    parser.add_argument('--lr_scheduler', type=bool, default=False, help='lr_scheduler - default False')
    parser.add_argument('--num_workers', type=int, default=11, help='number of workers - default 0')
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help='https://velog.io/@clayryu328/paper-review-CLIP-Learning-Transferable-Visual-Models-From-Natural-Language-Supervision')


    args = parser.parse_args()
    args.text_mode = [item for item in args.text_mode.split(',')]
    # sorry changed how to input text_mode it is now e.g. "title,ingredients"
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initializing tensorboard logger
    tb_logger        = TensorBoardLogger(save_dir = "tensorboard_logs", name=args.experiment_name)

    # Initializing the image encoder
    img_encoder      = get_image_encoder(args, device)

    # TODO: Recipe encoder should contain the concatenation technique within as well
    R_encoder        = get_text_encoder(args, device) 
    
    # Initializing the projection head
    projection_head  = get_head(args, device)

    # No negative pairs for ClipLoss
    if args.loss_fn == 'ClipLoss':
        args.p = 0.0

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

    # Fitting model
    trainer.fit(model = model) # , train_dataloaders = train_dataloader, val_dataloaders = val_dataloader)

    # Testing model
    trainer.test(model = model)
    # trainer.test(model = model,ckpt_path='tensorboard_logs/test/version_42/checkpoints/epoch=19-step=2980.ckpt')

    # title and instructions were used for this model
    # trainer.predict(model=model)
    # trainer.predict(model=model, ckpt_path='tensorboard_logs/test/version_52/checkpoints/epoch=1-step=296.ckpt')