import math
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import models as textmodels
from einops import rearrange, repeat
from transformers import AutoTokenizer, BertTokenizer, BertModel
import argparse

def to_device(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        ####################### insert code here ####################### 
        # self.k_projection = nn.Linear(embed_dim, embed_dim) # seems to also work
        # self.k_projection = nn.Linear(self.head_dim, self.head_dim) # seems to also work
        self.k_projection = nn.Linear(embed_dim, self.head_dim*self.num_heads)
        self.q_projection = nn.Linear(embed_dim, self.head_dim*self.num_heads)
        self.v_projection = nn.Linear(embed_dim, self.head_dim*self.num_heads)
        ################################################################
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size, seq_length, embed_dim = x.size()
        keys    = self.k_projection(x)
        queries = self.q_projection(x)
        values  = self.v_projection(x)

        # Rearrange keys, queries and values 
        # from batch_size x seq_length x embed_dim to (batch_size x num_head) x seq_length x head_dim
        keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        values = rearrange(values, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)

        ####################### insert code here ####################### 
        inner = queries @ keys.transpose(1, 2)
        inner *= self.scale
        attention = F.softmax(inner, dim=-1)
        out = attention @ values
        ################################################################

        # Rearragne output
        # from (batch_size x num_head) x seq_length x head_dim to batch_size x seq_length x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_length, seq_length)
        assert out.size() == (batch_size, seq_length, embed_dim)

        return self.o_projection(out)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.layernorm1(attention_out + x)
        x = self.dropout(x)
        fc_out = self.fc(x)
        x = self.layernorm2(fc_out + x)
        x = self.dropout(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0., max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, ::2]  = torch.sin(position * div_term)
        pe[:, 1::2]  = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x + self.pe[:, :seq_length]
        #return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super().__init__()
        # Your implemntation
        self.positional_embedding = nn.Parameter(torch.zeros(max_seq_len, embed_dim, device=device)).unsqueeze(0)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        x = x + self.positional_embedding[:, :seq_length]
        return x
        
def get_text_encoder(args, device:torch.device) -> nn.Module:
    
    class TransformerTextEmbedder(nn.Module):
        def __init__(self, embed_dim, num_heads, num_layers, max_seq_len,
                    pos_enc='fixed', pool='cls', dropout=0.0, 
                    fc_dim=None, num_tokens=50_000, embed_dim_out=64, 
                    
        ):
            super().__init__()

            assert pool in ['mean', 'max']
            assert pos_enc in ['fixed', 'learnable']
            

            self.pool, self.pos_enc, = pool, pos_enc
            self.token_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=num_tokens)

            if self.pos_enc == 'fixed':
                self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)
            elif self.pos_enc == 'learnable':
                self.positional_encoding = PositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_seq_len)

            transformer_blocks = []
            for i in range(num_layers):
                transformer_blocks.append(
                    EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

            self.transformer_blocks = nn.Sequential(*transformer_blocks)
            self.embedder = nn.Linear(embed_dim, embed_dim_out)
            self.output_dim = embed_dim_out
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, exp=False):

            tokens = self.token_embedding(x)
            batch_size, seq_length, embed_dim = tokens.size()

            x = self.positional_encoding(tokens)
            if exp:
                return x
            x = self.dropout(x)
            x = self.transformer_blocks(x)

            if self.pool =='max':
                x = x.max(dim=1)[0]
            elif self.pool =='mean':
                x = x.mean(dim=1)
                
            return self.embedder(x)
        
    class RobertaBase(nn.Module):
        def __init__(self, device:torch.device):            
            super(RobertaBase, self).__init__()

            self.device = device
            self.output_dim = 768
            
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
            self.model = BertModel.from_pretrained("bert-base-cased")
            for param in self.model.parameters(): 
                    param.requires_grad_(False)
                    
            # self.model.pooler = nn.Sequential(nn.Linear(768, self.output_dim)) 
            
            
        def forward(self, x) -> torch.Tensor:
            tokens = self.tokenizer(x, return_tensors = 'pt', padding=True)
            # we need to truncate the input sequence to 512 tokens due to limitations of the model
            max_sequence_length = 512
            for key, val in tokens.items():
                tokens[key] = tokens[key][:,:max_sequence_length]
            
            output = self.model(**tokens.to(self.device))
            last_hidden_state = output.last_hidden_state[:, 0, :]
            # output = self.model.pooler(last_hidden_state)            
            return last_hidden_state

    
    if args.text_encoder_name == 'transformer_base':
        return TransformerTextEmbedder(embed_dim=args.embedding_dim, 
                            num_heads=args.num_heads, 
                            num_layers=args.num_layers,
                            pos_enc=args.pos_enc,
                            pool=args.pool,  
                            dropout=args.dropout,
                            fc_dim=None,
                            max_seq_len=512, 
                            num_tokens=50_000, 
                            embed_dim_out=args.embedding_dim,
                            )
    if args.text_encoder_name == 'roberta_base':
        return RobertaBase(device=device)
    
def main(embed_dim=128, num_heads=4, num_layers=4, pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None, batch_size=2, embed_dim_out = 64):
        
    VOCAB_SIZE = 50_000
    SAMPLED_RATIO = 0.2
    MAX_SEQ_LEN = 512
    
    parser = argparse.ArgumentParser(description='Hello')
    
    parser.add_argument('--batch_size',    type=int, default=64, help='batch size - default 64')
    parser.add_argument('--embedding_dim', type=int, default=256, help='embedding dim - default 256')
    parser.add_argument('--margin', type=float, default=0.5, help='margin (for loss function) - default 0.5')
    parser.add_argument('--img_encoder_name', type=str, default="resnet", help='resnet, vit, efficientnet')
    parser.add_argument('--text_encoder_name', type=str, default="roberta_base", help='transformer_base, roberta_base')
    parser.add_argument("--text_mode", action="extend", nargs="+", type=str, default=['title'], help="text mode - default title")
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads - default 4')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs - default 20')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers - default 4')
    parser.add_argument('--pos_enc', type=str, default='fixed', help='positional encoding - default fixed')
    parser.add_argument('--pool', type=str, default='max', help='pooling - default max')
    parser.add_argument('--dropout', type=float, default=0.0, help='probability of dropout - default 0.0')


    args = parser.parse_args()
    
    model = get_text_encoder(args, device=device)

    if torch.cuda.is_available():
        model = model.to('cuda')

 #   from data.tokenization import train_dataloader
    from torch.utils.data import DataLoader, Dataset
    from data.dataloader import CombinedDataSet
    data_set = CombinedDataSet(p=0.2, mode='train', yield_raw_text=True)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    x = next(iter(data_loader))[1] # the text tuples
    print(model(x).shape) # [batch, embed_dim]
    
    
    
    

    # idxs, dummy_input = next(iter(train_dataloader))
    # dummy_input = dummy_input.to(device)
    # dummy_output = model(dummy_input)
    # print(dummy_output.shape)
    # print(dummy_output)

# def get_text_encoder(args, VOCAB_SIZE=50_000, SAMPLED_RATIO=0.2, MAX_SEQ_LEN=512):

#     model = TransformerTextEmbedder(embed_dim=args.embedding_dim, 
#                             num_heads=args.num_heads, 
#                             num_layers=args.num_layers,
#                             pos_enc=args.pos_enc,
#                             pool=args.pool,  
#                             dropout=args.dropout,
#                             fc_dim=None,
#                             max_seq_len=MAX_SEQ_LEN, 
#                             num_tokens=VOCAB_SIZE, 
#                             embed_dim_out=args.embedding_dim,
#                             )
    
#     if torch.cuda.is_available():
#         model = model.to('cuda')
    
#     return model



if __name__ == '__main__':
     main()
