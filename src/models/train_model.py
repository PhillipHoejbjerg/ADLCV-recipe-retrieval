import torch
import torch.nn as nn
import lightning as L
from lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
import argparse

class RecipeRetrievalLightningModule(L.LightningModule):
    def __init__(self, 
                 img_encoder, 
                 R_encoder, 
                 train_dataset, 
                 val_dataset, 
                 test_dataset,
                 loss_fn, 
                 lr = 0.001,
                 batch_size = 64,
                 embedding_dim = 256):
        
        super(RecipeRetrievalLightningModule, self).__init__()
        self.save_hyperparameters()

        self.loss_function = loss_fn
        self.img_encoder = img_encoder
        self.R_encoder = R_encoder
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset # TODO: IMPORTANT test dataloader should not have negative pairs!

        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # Mapping the output of the encoders to the embedding space
        self.W_R   = nn.Linear(R_encoder.output_dim, self.embedding_dim)
        self.W_img = nn.Linear(img_encoder.output_dim, self.embedding_dim)  

        # Accuracy
        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.test_dataset)) 

    def forward(self, img, R):
        
        # Getting latent space representations
        img_z, R_z = self.img_encoder(img), self.R_encoder(R)

        # Mapping to embedding space
        phi_img, phi_R = self.W_img(img_z), self.W_R(R_z)
        
        return phi_img, phi_R

    def training_step(self, batch, batch_idx):
        
        # Unpacking batch
        img, R, is_pos_pair = batch

        phi_img, phi_R = self(img, R)

        # Calculate loss here
        loss = self.loss_function(phi_img, phi_R, is_pos_pair)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # TODO: grab dataloader directly

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False) # TODO: grab dataloader directly
    
    # Entire test_set - in order to predict on entirety of test set
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False) # TODO: grab dataloader directly

    def validation_step(self, batch, batch_idx):
        
        # Unpacking batch
        img, R, is_pos_pair = batch

        phi_img, phi_R = self(img, R)

        # Calculate loss here
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
        R_acc   = self.accuracy(R_pred,   torch.arange(len(self.test_dataset)))
        img_acc = self.accuracy(img_pred, torch.arange(len(self.test_dataset)))

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

    args = parser.parse_args()

    # Initializing tensorboard logger
    tb_logger = TensorBoardLogger(save_dir = "tensorboard_logs", name=args.experiment_name)

    # Initializing the image encoder
    img_encoder = get_image_encoder("resnet")

    # TODO: Recipe encoder should contain the concatenation technique within as well
    R_encoder   = get_text_encoder() #default

    train_dataset = None
    val_dataset = None

    # Defining loss function
    loss_fn = torch.nn.CosineEmbeddingLoss(margin = args.margin, reduction='none')

    # Defining model
    model = RecipeRetrievalLightningModule(img_encoder, 
                                            R_encoder, 
                                            train_dataset, 
                                            val_dataset, 
                                            loss_fn, 
                                            batch_size = args.batch_size,
                                            embedding_dim = args.embedding_dim)

    # Defining callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    trainer = L.Trainer(max_steps=args.max_steps, 
                        default_root_dir="/models", # save_dir
                        callbacks=[checkpoint_callback],
                        logger = tb_logger)

    # Fitting model
    trainer.fit(model = model)

    # Testing model
    trainer.test(model = model)