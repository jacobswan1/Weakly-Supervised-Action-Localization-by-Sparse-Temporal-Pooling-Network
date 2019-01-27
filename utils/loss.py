import torch


def euclidean_distance(v1, v2, dim=None):
    if dim is not None:
        return torch.mean((v1 - v2) ** 2, dim)
    else:
        return torch.mean((v1 - v2) ** 2)


# Adaptive Margin Loss
class Adaptive_Margin_Loss(torch.nn.Module):
    """
    Adaptive margin loss penalize on the (pos, neg, lang) triples.
    Input:
    feat_pos:    (batch, 100)
    feat_neg:    (batch, # clips, 100)
    feat_lan:    (batch, 100)
    Refers to the document for equation.
    return loss1 + lambda*loss2
    """
    def __init__(self):
        super(Adaptive_Margin_Loss,self).__init__()

    def forward(self, feat_pos, feat_neg, feat_lan, device, margin=0.1, lamda=1):
        loss1 = euclidean_distance(feat_pos, feat_lan)
        loss2 = 0
        total = 0
        # Iterate through all batch
        for idx1, feats in enumerate(feat_neg):
            total += feats.shape[0]
            for idx2, feat in enumerate(feats):
                loss2 += torch.max(torch.zeros(1).to(device), margin-euclidean_distance(feat_pos[idx1], feat))
        loss2 /= total
        return loss1 + lamda*loss2[0]
