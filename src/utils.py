
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize

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
    def cosine_dist(im, s):
        """Cosine similarity between all the image and sentence pairs
        """

        return 1 - im.mm(s.t())


    def euclidean_dist(x, y):
        """
        Args:
            x: pytorch Variable, with shape [m, d]
            y: pytorch Variable, with shape [n, d]
        Returns:
            dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        return dist


    class TripletLoss(nn.Module):
        """Triplet loss class

        Parameters
        ----------
        margin : float
            Ranking loss margin
        metric : string
            Distance metric (either euclidean or cosine)
        """

        def __init__(self, margin=0.3, metric='cosine'):

            super().__init__()
            self.distance_function = euclidean_dist if metric == 'euclidean' else cosine_dist
            self.metric = metric
            self.margin = margin
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

        def forward(self, im, s):
            # compute image-sentence score matrix
            # batch_size x batch_size
            scores_i2r = self.distance_function(normalize(im, dim=-1),
                                                normalize(s, dim=-1))
            scores_r2i = scores_i2r.t()

            pos = torch.eye(im.size(0))
            neg = 1 - pos

            pos = (pos == 1).to(im.device)
            neg = (neg == 1).to(im.device)

            # positive similarities
            # batch_size x 1
            d1 = scores_i2r.diag().view(im.size(0), 1)
            d2 = d1.t()

            y = torch.ones(scores_i2r.size(0)).to(im.device)


            # image anchor - recipe positive bs x bs
            d1 = d1.expand_as(scores_i2r)
            # recipe anchor - image positive
            d2 = d2.expand_as(scores_i2r) #bs x bs

            y = y.expand_as(scores_i2r)

            # compare every diagonal score to scores in its column
            # recipe retrieval
            # batch_size x batch_size (each anchor is compared to all elements in the batch)
            cost_im = self.ranking_loss(scores_i2r, d1, y)
            # compare every diagonal score to scores in its row
            # image retrieval
            cost_s = self.ranking_loss(scores_i2r, d2, y)

            # clear diagonals
            cost_s = cost_s.masked_fill_(pos, 0)
            cost_im = cost_im.masked_fill_(pos, 0)

            return (cost_s + cost_im).mean()

    if args.loss_fn == 'contrastive':
        return ContrastiveLoss(margin = args.margin)

    if args.loss_fn == 'cosine':
        return nn.CosineEmbeddingLoss(margin = args.margin, reduction='mean')
    
    if args.loss_fn == 'ClipLoss':
        return ClipLoss()
    if args.loss_fn == 'TripletLoss':
        return TripletLoss(margin = args.margin, metric = 'cosine')

    
def denormalise_batch(imgs):
    '''denormalise a batch of images based on imagenet stats'''
    imgs = imgs.clone()
    imgs[:, 0, :, :] = imgs[:, 0, :, :] * 0.229 + 0.485
    imgs[:, 1, :, :] = imgs[:, 1, :, :] * 0.224 + 0.456
    imgs[:, 2, :, :] = imgs[:, 2, :, :] * 0.225 + 0.406
    return imgs
