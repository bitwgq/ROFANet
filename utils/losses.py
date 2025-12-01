from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss, DiceLoss, BCLoss
import torch.nn as nn
import torch
from torch.nn import functional as F

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

BCLoss = BCLoss()
DiceLoss = DiceLoss()


def hybrid_loss(predictions, targets):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    # focal = FocalLoss(gamma=0, alpha=None)

    for prediction, target in zip(predictions, targets):

        # bce = focal(prediction, target)
        # dice = dice_loss(prediction, target)
        # loss += bce + dice

        b_loss = BCLoss(prediction, target)
        d_loss = DiceLoss(prediction, target)
        # b_loss = BCLoss(prediction, target)
        loss += d_loss + b_loss

    return loss

class CriterionPC(nn.Module):
    def __init__(self, classes=2):
        super(CriterionPC, self).__init__()
        self.num_classes = classes

    def forward(self, preds, target):
        feat = preds
        feat = F.normalize(feat.view(-1, 16, 256*256), dim=1).view(-1, 16, 256, 256)
        size_f = (feat.shape[2], feat.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat.size())
        center_feat_S = feat.clone()

        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * center_feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        cos = nn.CosineSimilarity(dim=1)

        pcsim_feat = cos(feat, center_feat_S)

        center_feat = pcsim_feat.clone()

        for i in range(self.num_classes):
            mask_feat = (target == i).float()
            center_feat = (1 - mask_feat) * center_feat + mask_feat * ((mask_feat * pcsim_feat).sum(-1).sum(-1) / (mask_feat.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        weight = center_feat / (pcsim_feat + 1e-6)

        weight_mask = (weight > 1).float()

        weight_mask_one = (weight_mask == 0).float()

        weight = weight * weight_mask + weight_mask_one

        pcsim_feat = weight * pcsim_feat

        loss_only_change = pcsim_feat

        loss_only_change = loss_only_change.sum(-1).sum(-1)

        loss = - torch.log(loss_only_change)

        loss = loss.sum(-1) / 8

        return loss