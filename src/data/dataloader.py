import numpy as np
import random
import torch
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tokenization import yield_tokens_title, yield_tokens_title_and_ingredients, yield_tokens_title_and_ingredients_and_instructions, get_vocab
from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import ast

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

def denormalize(tensor):
    tensor = tensor * 0.225 + 0.45
    img = Image.fromarray(np.array(tensor.clamp(0, 1)*255,dtype=np.uint8))
    return img.resize((274, 169))
   
def parse_list_column(df):
    df.Cleaned_Ingredients = ast.literal_eval(df.Cleaned_Ingredients)
    df.Ingredients = ast.literal_eval(df.Ingredients)
    return df

def parse_datapipe(row):
    row[2] = ast.literal_eval(row[2])
    row[5] = ast.literal_eval(row[5])
    return row

class CombinedDataSet(Dataset):
    '''
    image, text, positive/negative pair (Bool)
    positive pair = the image matches the text
    parameter p, for probability of negative pair

    test set should not have negative pairs
    '''
    
    def __init__(self, p=0.2, mode='train', text=['title']):
        if text == ['title']:
            self.yield_tokens = yield_tokens_title
        elif text == ['title', 'ingredients']:
            self.yield_tokens = yield_tokens_title_and_ingredients
        elif text == ['title', 'ingredients', 'instructions']:
            self.yield_tokens = yield_tokens_title_and_ingredients_and_instructions
        
        self.path = 'data/processed/Food Images'
        self.image_files = os.listdir(self.path)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
        self.p = p

        FILE_PATH = 'data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
        print('building vocab')
        self.datapipe = IterableWrapper([FILE_PATH])
        self.datapipe = FileOpener(self.datapipe, mode='b')
        self.datapipe = self.datapipe.parse_csv(skip_lines=1, delimiter=',', quotechar='"', quoting=1)
        self.datapipe = self.datapipe.map(parse_datapipe)

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = get_vocab(self.datapipe, self.yield_tokens)

        self.text_transform = lambda x: [self.vocab['']] + [self.vocab[token] for token in self.tokenizer(x)] + [self.vocab['']]
        self.csv = pd.read_csv(FILE_PATH)
        self.csv.apply(parse_list_column, axis=1)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        if random.random() > self.p:
            _text = self.csv.iloc[idx]
            is_pos_pair = True
        else:
            idx =  torch.randint(0,self.__len__(),(1,)).item()
            _text = self.csv.iloc[idx]
            is_pos_pair = False
        text = _text.Title
        processed_text = torch.tensor(self.text_transform(text))

        image_path = 'data/processed/Food Images/' + _text.Image_Name + '.jpg'
        image = Image.open(image_path)
        image = self.transform(image)

        return image, processed_text, is_pos_pair

    
def collate_fn(batch):
    Image_Name, title, ingredients, instructions = zip(*batch)
    return Image_Name, title, ingredients, instructions

def collate_batch_text(batch):
    img, processed_text, is_positive = zip(*batch)
    text = pad_sequence(processed_text, padding_value=1.0, batch_first=True)
    return img, text, is_positive

def main(batch_size=2):

    data_set = CombinedDataSet(p=0.2, mode='train')
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch_text)
    fig, ax = plt.subplots(1,2, figsize=(14, 5))
    for img, text, is_positive in data_loader:
        print(text)
        for i in range(batch_size):
            title = data_set.vocab.lookup_tokens(list(text[i]))
            ax[i].imshow(denormalize(img[i].permute(1,2,0)))
            ax[i].set_title(' '.join(title))
        print(is_positive)

        plt.show()
        break

if __name__ == '__main__':
    main()