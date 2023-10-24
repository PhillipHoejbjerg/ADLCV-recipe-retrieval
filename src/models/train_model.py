import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
import lightning as L

MARGIN = 0.1

class RecipeRetrieval(L.LightningModule):
    def __init__(self, img_encoder, R_encoder, loss_function):
        super().__init__()
        self.img_encoder   = img_encoder
        self.R_encoder     = R_encoder
        self.loss_function = loss_function

    def training_step(self, batch, batch_idx):

        R, img, is_pos_pair = batch # (Recipe, img, positive pair)

        img = img.view(img.size(0), -1)

        img_z = self.img_encoder(img)
        R_z   = self.R_encoder(R)

        loss = loss_function(img_z, R_z, is_pos_pair)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
img_encoder = None
R_encoder = None    
loss_function = torch.nn.CosineEmbeddingLoss(margin = MARGIN, reduction='none')

trainer = L.Trainer(max_steps=1000)
trainer.fit(RecipeRetrieval(img_encoder, R_encoder, loss_function), data.DataLoader(?, batch_size=64))