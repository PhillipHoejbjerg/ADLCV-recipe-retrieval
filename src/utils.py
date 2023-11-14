
import torch.nn as nn
import torch
import torch.nn.functional as F


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
        def __init__(self, W_R, W_img):
            super().__init__()
            self.W_R = W_R
            self.W_img = W_img

        def forward(self, I_emb, T_emb):
            I_e = I_emb @ self.W_img / torch.norm(I_emb, dim=1, keepdim=True)
            T_e = T_emb @ self.W_R / torch.norm(T_emb, dim=1, keepdim=True)
            # TODO: t is temperature
            logits = I_e @ T_e.T #* torch.exp(t)

            labels = torch.arange(logits.shape[0]).to(logits.device)
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            loss = (loss_i + loss_t) / 2
            return torch.mean(loss)


    if args.loss_fn == 'contrastive':
        return ContrastiveLoss(margin = args.margin)

    if args.loss_fn == 'cosine':
        return nn.CosineEmbeddingLoss(margin = args.margin, reduction='mean')
    
    if args.loss_fn == 'clip':
        return ClipLoss


    # TODO: Figure out how to do positive pairs with no mulitples of labels    
    if args.loss_fn == 'triplet':
        return nn.TripletMarginLoss(margin = args.margin, reduction='mean')

    
