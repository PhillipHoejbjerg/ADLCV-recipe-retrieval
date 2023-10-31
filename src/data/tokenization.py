import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# https://github.com/andrei-radulescu-banu/stat453-deep-learning-ss21/blob/main/L15/migration_tutorial.ipynb

FILE_PATH = 'data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'

datapipe = IterableWrapper([FILE_PATH])
datapipe = FileOpener(datapipe, mode='b')
datapipe = datapipe.parse_csv(skip_lines=1)

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, Title, Ingredients, Instructions, Image_Name, Cleaned_Ingredients in data_iter:
        yield tokenizer(Title)  # , tokenizer(Ingredients), tokenizer(Instructions), tokenizer(Image_Name), tokenizer(Cleaned_Ingredients)

def get_vocab(train_datapipe):
    # TODO: we might not need the special tokens
    vocab = build_vocab_from_iterator(yield_tokens(train_datapipe),
                                      specials=['<UNK>', '<PAD>'],
                                      max_tokens=20000)
    vocab.set_default_index(vocab['<UNK>'])
    return vocab

vocab = get_vocab(datapipe)

def collate_batch(batch):
    label_list, text_list = [], []
    text_transform = lambda x: [vocab['']] + [vocab[token] for token in tokenizer(x)] + [vocab['']]

    for _text in batch:
        # label_list.append(label_transform(_label))
        text = _text[1] # just the title for now
        label = int(_text[0])
        processed_text = torch.tensor(text_transform(text))
        text_list.append(processed_text)
        label_list.append(label)

    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

train_iter = datapipe
train_dataloader = DataLoader(list(train_iter),
                              batch_size=8,
                              shuffle=True,
                              collate_fn=collate_batch)


train_iter = datapipe
train_list = list(train_iter)
batch_size = 8  # A batch size of 8

class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset, batch_size, indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indices and length
        self.indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(dataset)]
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        batches = [self.pooled_indices[i:i + self.batch_size] for i in
                range(0, len(self.pooled_indices), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size
  
bucket_dataloader = DataLoader(train_list,
                               batch_sampler=BatchSamplerSimilarLength(
                                     dataset = train_list, 
                                     batch_size=batch_size),
                               collate_fn=collate_batch)

if __name__ == '__main__':
    print(next(iter(bucket_dataloader)))

    print(next(iter(train_dataloader)))
