import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from src.data.dataloader import get_dataloader

from src.models.ImageEncoder import get_image_encoder
from src.models.text_encoder import get_text_encoder # TODO: Does not exist yet
from src.utils import get_loss_fn

from src.models.train_model import RecipeRetrievalLightningModule

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recipe Retrieval Training Script')
    
    parser.add_argument('--batch_size',    type=int, default=64, help='batch size - default 64')
    parser.add_argument('--embedding_dim', type=int, default=256, help='embedding dim - default 256')
    parser.add_argument('--margin', type=float, default=0.5, help='margin (for loss function) - default 0.5')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate - default 0.001')
    parser.add_argument('--experiment_name', type=str, default="test", help='Experiment name - default test')
    parser.add_argument('--img_encoder_name', type=str, default="resnet", help='resnet, vit, efficientnet')
    parser.add_argument('--loss_fn', type=str, default="cosine", help='Loss_fn - default cosine')
    parser.add_argument('--p', type=float, default=0.8, help='probability of negative pair - default 0.8')
    parser.add_argument("--text_mode", action="extend", nargs="+", type=str, default=['title'], help="text mode - default title")
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads - default 4')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs - default 20')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers - default 4')
    parser.add_argument('--pos_enc', type=str, default='fixed', help='positional encoding - default fixed')
    parser.add_argument('--pool', type=str, default='max', help='pooling - default max')
    parser.add_argument('--dropout', type=float, default=0.0, help='probability of dropout - default 0.0')


    args = parser.parse_args()

    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initializing tensorboard logger
    tb_logger        = TensorBoardLogger(save_dir = "tensorboard_logs", name=args.experiment_name)

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

    trainer = L.Trainer(default_root_dir="/models", # save_dir
                        callbacks=[checkpoint_callback],
                        logger = tb_logger,
                        max_epochs=args.num_epochs)

    # Testing model
    trainer.predict(model = model, ckpt_path='tensorboard_logs/contrastive/version_0/checkpoints/epoch=15-step=2384.ckpt')