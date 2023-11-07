
import torch.nn as nn
import torch
import torch.nn.functional as F


def get_loss_fn(args):

    # ChatGPT implementation : check if properly working
    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=0.2):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, output1, output2, target):
            # Calculate cosine similarity
            cosine_similarity = F.cosine_similarity(output1, output2)

            # Calculate loss using contrastive loss formula
            loss = 0.5 * (1 - target) * cosine_similarity**2 + \
                0.5 * target * torch.clamp(self.margin - cosine_similarity, min=0)

            return loss.mean()
        
    if args.loss_fn == 'contrastive':
        return ContrastiveLoss(margin = args.margin)

    if args.loss_fn == 'cosine':
        return nn.CosineEmbeddingLoss(margin = args.margin, reduction='none')

    # TODO: Figure out how to do positive pairs with no mulitples of labels    
    if args.loss_fn == 'triplet':
        return nn.TripletMarginLoss(margin = args.margin, reduction='none')

    