
import torch.nn as nn
import torch
import torch.nn.functional as F

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def get_loss_fn(args):

    # ChatGPT implementation : check if properly working
    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=0.2):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, embed1, embed2, is_pos_pair):
            # Calculate cosine similarity
            cosine_similarity = F.cosine_similarity(embed1, embed2)

            # Calculate loss using contrastive loss formula
            loss = 0.5 * is_pos_pair.float() * cosine_similarity**2 + \
                0.5 * (1 - is_pos_pair.float()) * torch.clamp(self.margin - cosine_similarity, min=0)**2

            return loss.mean()
        
    class ClipLoss(nn.Module):
        '''figure 3 from clip paper and 
        https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2'''
        def __init__(self):
            super().__init__()

        def forward(self, I_emb, R_emb, temperature):
            # Calculating the Loss
            logits = (R_emb @ I_emb.T) / temperature
            I_similarity = I_emb @ I_emb.T
            R_similarity = R_emb @ R_emb.T
            targets = F.softmax(
                (I_similarity + R_similarity) / 2 * temperature, dim=-1
            )
            R_loss  = cross_entropy(logits, targets, reduction='none')
            I_loss  = cross_entropy(logits.T, targets.T, reduction='none')
            loss =  (I_loss + R_loss) / 2.0 # shape: (batch_size)            

            return loss.mean()

    if args.loss_fn == 'contrastive':
        return ContrastiveLoss(margin = args.margin)

    if args.loss_fn == 'cosine':
        return nn.CosineEmbeddingLoss(margin = args.margin, reduction='mean')
    
    if args.loss_fn == 'ClipLoss':
        return ClipLoss()


    # TODO: Figure out how to do positive pairs with no mulitples of labels    
    if args.loss_fn == 'triplet':
        return nn.TripletMarginLoss(margin = args.margin, reduction='mean')

    
