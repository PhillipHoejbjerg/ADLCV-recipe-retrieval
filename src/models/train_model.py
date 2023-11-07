import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from src.data.dataloader import get_dataloader

from src.models.ImageEncoder import get_image_encoder
from src.models.text_encoder import get_text_encoder # TODO: Does not exist yet
from src.utils import get_loss_fn

torch.set_float32_matmul_precision('high')

class RecipeRetrievalLightningModule(L.LightningModule):
    def __init__(self, 
                 img_encoder, 
                 R_encoder, 
                 train_dataloader, 
                 val_dataloader, 
                 test_dataloader,
                 loss_fn, 
                 lr = 0.001,
                 batch_size = 64,
                 embedding_dim = 256):
        
        super().__init__()
        self.save_hyperparameters(ignore=['img_encoder', 'R_encoder', 'loss_fn'])

        self.loss_function      = loss_fn
        self.img_encoder        = img_encoder
        self.R_encoder          = R_encoder
        self.train_dataloader_  = train_dataloader
        self.val_dataloader_    = val_dataloader
        self.test_dataloader_   = test_dataloader # TODO: IMPORTANT test dataloader should not have negative pairs!

        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # Mapping the output of the encoders to the embedding space
        self.W_R   = nn.Linear(R_encoder.output_dim,   self.embedding_dim)
        self.W_img = nn.Linear(img_encoder.output_dim, self.embedding_dim)  

        # Accuracy
        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.test_dataloader_)) # TODO: Get len from test dataset

    def forward(self, img, R):
        
        # Getting latent space representations
        img_z, R_z = self.img_encoder(img), self.R_encoder(R)

        # Mapping to embedding space
        phi_img, phi_R = self.W_img(img_z), self.W_R(R_z)
        
        return phi_img, phi_R

    def training_step(self, batch, batch_idx):
        
        # Unpacking batch
        img, R, is_pos_pair = batch

        phi_img, phi_R = self.forward(img, R)

        # Calculate loss here
        loss = self.loss_function(phi_img, phi_R, is_pos_pair)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_ 
    
    # Entire test_set - in order to predict on entirety of test set
    def test_dataloader(self):
        return self.test_dataloader_ 
    
    def validation_step(self, batch, batch_idx):
        
        # Unpacking batch
        img, R, is_pos_pair = batch

        # Getting latent space representations
        phi_img, phi_R = self(img, R)

        # Calculate loss
        loss = self.loss_function(phi_img, phi_R, is_pos_pair)
        
        self.log("val_loss", loss)

    # TODO: Also do this for prediction
    def test_step(self, batch, batch_idx):

        # Unpacking batch
        img, R, _ = batch # TODO: IMPORTANT test dataloader should not have negative pairs!

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

        # Calculating accuracy
        R_acc   = self.accuracy(R_pred,   torch.arange(len(self.test_dataloader_)))
        img_acc = self.accuracy(img_pred, torch.arange(len(self.test_dataloader_)))

        # Logging metrics
        metrics = {"img_acc": img_acc, "R_acc": R_acc}
        self.log_dict(metrics)

    def predict_step(self, batch, batch_idx):

        # Unpacking batch
        img, R, _ = batch # TODO: IMPORTANT test dataloader should not have negative pairs!

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

        return R_pred, img_pred

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recipe Retrieval Training Script')
    
    parser.add_argument('--batch_size',    type=int, default=64, help='batch size - default 64')
    parser.add_argument('--embedding_dim', type=int, default=256, help='embedding dim - default 256')
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps - default 1000')
    parser.add_argument('--margin', type=float, default=0.5, help='margin (for loss function) - default 0.5')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate - default 0.001')
    parser.add_argument('--experiment_name', type=str, default="test", help='Experiment name - default test')
    parser.add_argument('--img_encoder_name', type=str, default="resnet", help='Model name - default resnet')
    parser.add_argument('--loss_fn', type=str, default="cosine", help='Loss_fn - default cosine')
    parser.add_argument('--p', type=float, default=0.8, help='probability of negative pair - default 0.8')
    parser.add_argument("--text_mode", action="extend", nargs="+", type=str, default=['title'], help="text mode - default title")
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads - default 4')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers - default 4')
    parser.add_argument('--pos_enc', type=str, default='fixed', help='positional encoding - default fixed')
    parser.add_argument('--pool', type=str, default='max', help='pooling - default max')
    parser.add_argument('--dropout', type=float, default=0.0, help='probability of dropout - default 0.0')


    args = parser.parse_args()

    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initializing tensorboard logger
    tb_logger = TensorBoardLogger(save_dir = "tensorboard_logs", name=args.experiment_name)

    # Initializing the image encoder
    img_encoder      = get_image_encoder(args, device)

    # TODO: Recipe encoder should contain the concatenation technique within as well
    R_encoder        = get_text_encoder(args)

    train_dataloader = get_dataloader(args, mode = 'train')
    val_dataloader   = get_dataloader(args, mode = 'val')
    test_dataloader  = get_dataloader(args, mode = 'test')

    # Defining loss function
    loss_fn = get_loss_fn(args)

    # Defining model
    model = RecipeRetrievalLightningModule(img_encoder, 
                                            R_encoder, 
                                            train_dataloader, 
                                            val_dataloader, 
                                            test_dataloader,
                                            loss_fn, 
                                            lr = args.lr,
                                            batch_size = args.batch_size,
                                            embedding_dim = args.embedding_dim)
    

    # Defining callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    trainer = L.Trainer(max_steps=args.max_steps, 
                        default_root_dir="/models", # save_dir
                        callbacks=[checkpoint_callback],
                        logger = tb_logger)

    # Fitting model
    trainer.fit(model = model) # , train_dataloaders = train_dataloader, val_dataloaders = val_dataloader)

    # Testing model
    trainer.test(model = model)
