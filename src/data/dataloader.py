import numpy as np
import random
import torch
# use torchtext v.0.6.0
from torchtext import data, datasets, vocab
import torchdata.datapipes as dp
# import torchtext.transforms as T
import torchvision
import torchvision.transforms as T
from torchtext.vocab import build_vocab_from_iterator
import os
from PIL import Image


NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def to_device(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class ImageDataLoader(torch.utils.data.DataLoader):
    def __init__(self):
        self.path = 'data/processed/Food Images'
        self.image_files = os.listdir(self.path)
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.image_files[idx])
        image = Image.open(image_path)
        return image

class ContrastiveImageDataLoader(torch.utils.data.DataLoader):
    def __init__(self):
        self.path = 'data/processed/Food Images'
        self.image_files = os.listdir(self.path)
    def __len__(self):
        return len(self.image_files)
    def random_idx(self, idx):
        r_idx =  torch.randint(0,len(self.image_files),1).item()
        if r_idx == idx:
            r_idx = self.random_idx(idx)
        return r_idx
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.image_files[idx])
        image = Image.open(image_path)
        r_idx = self.random_idx(idx)
        contrast_img = os.path.join(self.path, self.image_files[r_idx])
        contrast_img = Image.open(contrast_img)
        return image, contrast_img
    
class TextDataLoader(torch.utils.data.DataLoader):
    def __init__(self):
        FILE_PATH = 'data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
        self.data_pipe = dp.iter.IterableWrapper([FILE_PATH])
        self.data_pipe = dp.iter.FileOpener(self.data_pipe, mode='rb')
        self.data_pipe = self.data_pipe.parse_csv(skip_lines=1, delimiter=',', as_tuple=True)
        # Image_Name = data_pipe[4]
        # Recipe_title = data_pipe[1]
        self.csv = list(self.data_pipe)
        self.tokens = []
    def __len__(self):
        return len(self.csv)
    
    def tokenize(self, text):
        return text.split()
    
    def __getitem__(self, idx):
        text = self.data_pipe[idx]
        Image_Name = text[4]
        Recipe_title = text[1]
        tokens = self.tokenize(Recipe_title)
        return tokens, Image_Name
    
def main(batch_size=16):
    data_loader = TextDataLoader()
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=True)
    for tokens, Image_Name in data_loader:
        print(tokens)
        print(Image_Name)
        break

if __name__ == '__main__':
    main()