import numpy as np
import random
import torch
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from src.data.tokenization import yield_tokens_title, yield_tokens_title_and_ingredients, yield_tokens_title_and_ingredients_and_instructions, get_vocab
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
    
    def __init__(self, p=0.8, mode='train', text=['title'], yield_raw_text=False, args=None):
        assert mode in ['train', 'test', 'val']
        self.text_mode = text
        if self.text_mode == ['title']:
            self.yield_tokens = yield_tokens_title
        elif self.text_mode == ['title', 'ingredients']:
            self.yield_tokens = yield_tokens_title_and_ingredients
        elif self.text_mode == ['title', 'ingredients', 'instructions']:
            self.yield_tokens = yield_tokens_title_and_ingredients_and_instructions

        self.path = 'data/processed/Food Images'
        self.image_paths = os.listdir(self.path)
        # Compute splits sizes
        # this assumes that the data is shuffled beforehand

        
        self.transform = T.Compose([
            T.CenterCrop(224) if args.center_crop else T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
        self.p = p
        self.yield_raw_text = yield_raw_text

        FILE_PATH = 'data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'

        if not self.yield_raw_text:
            print('building vocab')
            self.datapipe = IterableWrapper([FILE_PATH])
            self.datapipe = FileOpener(self.datapipe, mode='b')
            self.datapipe = self.datapipe.parse_csv(skip_lines=1, delimiter=',', quotechar='"', quoting=1)
            self.datapipe = self.datapipe.map(parse_datapipe)
            self.tokenizer = get_tokenizer('basic_english')
            self.vocab = get_vocab(self.datapipe, self.yield_tokens)

            self.text_transform = lambda x: [self.vocab['']] + [self.vocab[token] for token in self.tokenizer(x)] + [self.vocab['']]
        
        self.csv = pd.read_csv(FILE_PATH)
        # Shuffling csv
        self.csv = self.csv.sample(frac=1, random_state=42).reset_index(drop=True)

        # Get dataset split
        # Since the text data set is used to load a corresponding image, we just select on the csv
        train_size = int(0.7 * len(self.csv))
        val_size   = int(0.1 * len(self.csv))

        if mode == 'train':
            self.csv = self.csv.iloc[:train_size, :]
        elif mode == 'val':
            self.csv = self.csv.iloc[train_size:train_size+val_size, :]
        elif mode == 'test':
            self.csv = self.csv.iloc[train_size+val_size:, :]
            self.p = 0
        self.csv = self.csv.apply(parse_list_column, axis=1)

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        
        if random.random() > self.p:
            _text = self.csv.iloc[idx]
            is_pos_pair = True
        else:
            idx =  torch.randint(0,self.__len__(),(1,)).item()
            _text = self.csv.iloc[idx]
            is_pos_pair = False

        # Building text feature
        text = None
        if 'title' in self.text_mode:
            text += _text.Title
        if 'ingredients' in self.text_mode:
            text += ' ' + ' '.join(str(e) for e in _text.Cleaned_Ingredients)
        if 'instructions' in self.text_mode:
            text += ' ' + _text.Instructions

        image_path = 'data/processed/Food Images/' + _text.Image_Name + '.jpg'
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        if self.yield_raw_text:
            # Using bert-base-uncased we need to lower txt
            return image, text.lower(), is_pos_pair
        else:
            processed_text = torch.tensor(self.text_transform(text))
            return image, processed_text, is_pos_pair

    
def collate_fn(batch):
    Image_Name, title, ingredients, instructions = zip(*batch)
    return Image_Name, title, ingredients, instructions

def collate_batch_text(batch):
    img, processed_text, is_positive = zip(*batch)
    text = pad_sequence(processed_text, padding_value=1.0, batch_first=True)
    img = torch.stack(img)
    return img, text, torch.tensor(is_positive)
    
def collate_batch_text_roberta(batch):
    img, processed_text, is_positive = zip(*batch)
    img = torch.stack(img)
    return img, list(processed_text), torch.tensor(is_positive)    

def main(batch_size=2):
    # how to get raw text from dataloaders
    data_set = CombinedDataSet(p=0, mode='train',text=['title', 'instructions'], yield_raw_text=True)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True,collate_fn=collate_batch_text_roberta)
    # print(next(iter(data_loader))[0].shape)
    # print(next(iter(data_loader))[1])
    fig, ax = plt.subplots(1,2, figsize=(14, 5))
    for img, text, is_positive in data_loader:
        for i in range(batch_size):
            title = text[i]
            ax[i].imshow(denormalize(img[i].permute(1,2,0)))
            ax[i].set_title(title)
            print(title)
        print(is_positive)
        plt.savefig('reports/figures/data_ex.png', dpi=300, bbox_inches='tight')
        plt.show()
        break # just one batch

    data_set = CombinedDataSet(p=0.2, mode='test', text=['title', 'ingredients'])
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch_text)
    fig, ax = plt.subplots(1,2, figsize=(14, 5))
    for img, text, is_positive in data_loader:
        # print(text)
        for i in range(batch_size):
            title = data_set.vocab.lookup_tokens(list(text[i]))
            ax[i].imshow(denormalize(img[i].permute(1,2,0)))
            ax[i].set_title(' '.join(title))
        # print(is_positive)

        plt.show()
        break

def get_dataloader(args, mode = 'train'):
    
    if args.text_encoder_name == 'roberta_base':
        coll = collate_batch_text_roberta
        yield_raw_text = True
    else:
        coll = collate_batch_text
        yield_raw_text = False

    # No negative pairs for ClipLoss
    if args.loss_fn == 'ClipLoss' or args.loss_fn == 'TripletLoss':
        args.p = 0.0
	
    data_set  = CombinedDataSet(p=0.0, mode=mode, text=args.text_mode, yield_raw_text=yield_raw_text, args=args) if mode == 'test' else CombinedDataSet(p=args.p, mode=mode, text=args.text_mode, yield_raw_text=yield_raw_text, args=args)

    # Dictionary of parameters for each mode
    mode_dict = {'train': {'batch_size': args.batch_size, 'shuffle': True},
                 'val':   {'batch_size': args.batch_size, 'shuffle': False},
                 'test':  {'batch_size': 250,            'shuffle': False}}
                # we need to subsample the test set to fit in memory
    data_loader = DataLoader(data_set, batch_size=mode_dict[mode]['batch_size'], shuffle=mode_dict[mode]['shuffle'], collate_fn=coll, num_workers=args.num_workers)

    return data_loader

if __name__ == '__main__':
    
    main()
