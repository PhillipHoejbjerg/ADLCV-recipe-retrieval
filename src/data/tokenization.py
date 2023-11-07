from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# https://github.com/andrei-radulescu-banu/stat453-deep-learning-ss21/blob/main/L15/migration_tutorial.ipynb

tokenizer = get_tokenizer('basic_english')

def yield_tokens_title(data_iter):
    for idx, Title, Ingredients, Instructions, Image_Name, Cleaned_Ingredients in data_iter:
        yield tokenizer(Title)
        
def yield_tokens_title_and_ingredients(data_iter):
    for idx, Title, Ingredients, Instructions, Image_Name, Cleaned_Ingredients in data_iter:
        a = Title + ' ' + ' '.join(str(e) for e in Cleaned_Ingredients)
        yield tokenizer(a)

def yield_tokens_title_and_ingredients_and_instructions(data_iter):
    for idx, Title, Ingredients, Instructions, Image_Name, Cleaned_Ingredients in data_iter:
        a = Title + ' ' + ' '.join(str(e) for e in Cleaned_Ingredients)
        b = a + ' ' + Instructions
        yield tokenizer(b)

def get_vocab(train_datapipe, yield_tokens):
    # TODO: we might not need the special tokens
    vocab = build_vocab_from_iterator(yield_tokens(train_datapipe),
                                      specials=['<EOS>','<PAD>'],
                                      max_tokens=20000)
    vocab.set_default_index(vocab['<EOS>'])
    return vocab

