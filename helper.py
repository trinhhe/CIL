import torch
import torch.nn.functional as f
from PIL import Image
import os
import numpy as np
import torchvision as tv
import torchvision.transforms.functional as trans
from torch.utils import data


#https://pytorch.org/vision/stable/_modules/torchvision/models/segmentation/segmentation.html
def getModel(modelName):
    if modelName== 'mobile':
        model = tv.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
    elif modelName == 'deeplab101':
        model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
    elif modelName == 'deeplab50':
        model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))

    for param in model.parameters():
            param.requires_grad = True

    return model.cuda()

def cross_entropy(inp, tar, reduction = 'sum'):
    x = inp.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
    y = torch.squeeze(tar).view(-1)
    return f.cross_entropy(x, y, reduction=reduction)

class ImageLoader(data.Dataset):
    def __init__(self, testDir = None, imgDirTrain = None, gtDirTrain = None):
        self.forTraining = False
        if imgDirTrain:
            self.forTraining = True
            imgDir = imgDirTrain
            self.gtDirs = np.asarray([os.path.join(gtDirTrain, filename) for filename in os.listdir(gtDirTrain)])
        else:
            imgDir = testDir
        self.ids = np.asarray([os.path.join(imgDir, filename)for filename in os.listdir(imgDir)])
   
    def augment(self, image, gt):
        #https://pytorch.org/vision/stable/transforms.html
        augmentor = tv.transforms.Compose([
            tv.transforms.RandomGrayscale(0.1),
            tv.transforms.ColorJitter(0.25, 0.26, 0.22, 0.24),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        image = augmentor(image)
        
        gt = trans.to_tensor(gt)
        gt = torch.where(gt < 0.7, torch.zeros_like(gt), torch.ones_like(gt))
        gt = gt.to(torch.int64)
        return image, gt

    def __getitem__(self, index):
        image = Image.open(self.ids[index])
        _item_ = dict()
        if not self.forTraining:
            # https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
            normalize = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            normalized = normalize(image)
            _item_.update([("image", normalized)])
            # print(_item_["image"].size)
            return _item_

        gt = Image.open(self.gtDirs[index])
        image, gt = self.augment(image, gt)

        _item_.update([("gt", gt),("image", image)])
        return _item_

    def __len__(self):
        return len(self.ids)

