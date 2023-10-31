import torch.nn as nn, torchvision as tv
import torch

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


    
