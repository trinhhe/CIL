from numpy.lib.function_base import gradient
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
# Jaccard Loss, refered from: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return 1 - IoU

class GradientLoss(nn.Module):
    
    def __init__(self):
        super(GradientLoss ,self).__init__()

    def gradient(self, x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        # h_x = x.size()[-2]
        # w_x = x.size()[-1]
        
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx= right - left
        dy= bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def forward(self , inputs , targets):
        inputs = inputs.reshape([inputs.shape[0],1,inputs.shape[1],inputs.shape[2]])
        dx , dy = self.gradient(inputs)
        lap_grad_norm = torch.norm(dx) + torch.norm(dy)

        return lap_grad_norm

        # return F.l1_loss(inputs, targets)


class MultiLoss(nn.Module):
    def __init__(self ,  loss_list = [], weights = None ):
        super(MultiLoss,self).__init__()

        self.losses = []
        self.weights = weights
        for loss in loss_list:
            self.losses.append(eval(loss)())

    def forward(self, inputs , targets , smooth = 1):

        total_loss = 0
        for loss , weight in zip(self.losses , self.weights):
            total_loss += loss(inputs , targets)*weight
        
        return total_loss