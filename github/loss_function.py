import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class cross_entropy(nn.Module):
    def __init__(self, weight=None):
        super(cross_entropy, self).__init__()
        self.loss = nn.NLLLoss(weight)
    
    def forward(self, outputs, labels):
        labels = labels.long()
        loss = self.loss(F.log_softmax(outputs, dim=1),labels)
        return loss


