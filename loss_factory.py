from pywick import losses as ls
import torch
import torch.nn as nn
from configs import *

class DiceFocalLoss(nn.Module):
    '''
        :param num_classes: number of classes
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                            focus on hard misclassified example
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        :param weights: (list(), default = [1,1]) Optional weighing (0.0-1.0) of the losses in order of [dice, focal]
    '''
    def __init__(self, focal_param, weights=[1.0,1.0], **kwargs):
        super(DiceFocalLoss, self).__init__()
        self.dice = ls.SoftDiceLoss()
        self.focal = ls.BinaryFocalLoss(gamma=focal_param)
        self.weights = weights

    def forward(self, logits, targets):
        #return self.dice(logits, targets)
        #return self.focal(logits, targets)
        return self.weights[0] * self.dice(logits, targets) + \
                self.weights[1] * self.focal(logits, targets)

class BCE_Tversky(nn.Module):
    def __init__(self, weights=[1.0, 1.0], **kwargs):
        super(BCE_Tversky, self).__init__()
        self.bce = ls.BCELoss2d()
        self.tversky = ls.TverskyLoss(alpha=0.5, beta=0.7)
        self.weights = weights

    def forward(self, logits, targets):
        return self.weights[0] * self.bce(logits, targets) + \
                self.weights[1] * self.tversky(logits, targets)


class BCEdicepenalizeborder_Tversky(nn.Module):
    def __init__(self, weights=[1.0, 1.0], **kwargs):
        super(BCEdicepenalizeborder_Tversky, self).__init__()
        self.bce_dice = ls.BCEDicePenalizeBorderLoss()
        self.tversky = ls.TverskyLoss(alpha=0.5, beta=0.7)
        self.weights = weights

    def forward(self, logits, targets):
        return self.weights[0] * self.bce_dice(logits, targets) + \
                self.weights[1] * self.tversky(logits, targets)


class Loss_Factory:
    @staticmethod
    def get_loss(loss='bce'):
        if loss == 'bce':
            return ls.BCELoss2d()
        elif loss == 'dice':
            return ls.SoftDiceLoss() 
        elif loss == 'focal':
            return ls.BinaryFocalLoss(gamma=0)
        elif loss == 'bce_dice':
            return ls.BCEDiceLoss()
        elif loss == 'bce_dice_focal':
            return ls.BCEDiceFocalLoss(focal_param=0.5)
        elif loss == 'dice_focal':
            return DiceFocalLoss(focal_param=focal_param, weights=[1, 0])
        elif loss == "bcedicepenalizeborderloss":
            return ls.BCEDicePenalizeBorderLoss()
        elif loss == "lovaszsoftmax":
            return ls.LovaszSoftmax()
        elif loss == "activecontourloss":
            return ls.ActiveContourLoss()
        elif loss == "tverskyloss":
            return ls.TverskyLoss(alpha=0.5, beta=0.7)
        elif loss == "focalbinarytverskyloss":
            return ls.FocalBinaryTverskyLoss()
        elif loss == "poissonloss":
            return ls.PoissonLoss()
        elif loss == "combobcediceloss":
            return ls.ComboBCEDiceLoss()
        elif loss == "bce_tversky":
            return BCE_Tversky()
        elif loss == "bcedicepenalizeborder_tversky":
            return BCEdicepenalizeborder_Tversky()

