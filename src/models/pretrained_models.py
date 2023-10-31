import torch.nn as nn, torchvision as tv
import torch
from text_encoder import TransformerTextEmbedder
from torch.utils.data import DataLoader

def get_image_encoder(model = 'resnet50'):
    
    if model == 'resnet50':
        # Load the pre-trained model
        img_encoder = tv.models.resnet50(pretrained=True)

        # Remove the last layer
        img_encoder = nn.Sequential(*list(img_encoder.children())[:-1])

    return img_encoder


class IngredientsEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(IngredientsEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)

    def forward(self, x):

        # Not needed as default is zero anyways
        h0 = torch.zeros(self.num_layers*2 if self.bidirectional else self.num_layers, x.size(1), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers*2 if self.bidirectional else self.num_layers, x.size(1), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        return out[-1, :, :]

def get_ingredients_encodr(model = 'bidirectional_lstm'):

    if model == 'bidirectional_lstm':

        return IngredientsEncoder(input_size=300, hidden_size=512, num_layers=2, bidirectional=True)


def get_text_encoder(embed_dim=512, num_heads=8, num_layers=6, pos_enc='fixed', pool='mean', dropout=0.1, fc_dim=None, embed_dim_out=128):
    VOCAB_SIZE = 50_000
    MAX_SEQ_LEN = 512
    model = TransformerTextEmbedder(embed_dim=embed_dim, num_heads=num_heads, 
                                num_layers=num_layers,
                                pos_enc=pos_enc,
                                pool=pool,  
                                dropout=dropout,
                                fc_dim=fc_dim,
                                max_seq_len=MAX_SEQ_LEN, 
                                num_tokens=VOCAB_SIZE, 
                                embed_dim_out=embed_dim_out,
                                )
    return model


if __name__ == '__main__':
    get_text_encoder(text=['title', 'ingredients', 'instructions'])
