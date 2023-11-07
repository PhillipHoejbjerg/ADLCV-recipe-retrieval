
import torch.nn as nn

def get_loss_fn(args):
    
    if args.loss_fn == 'cosine':
        return nn.CosineEmbeddingLoss(margin = args.margin, reduction='none')

    # TODO: Figure out how to do positive pairs with no mulitples of labels    
    if args.loss_fn == 'triplet':
        return nn.TripletMarginLoss(margin = args.margin, reduction='none')

    