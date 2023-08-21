from unet import UNet, UNetBranched
from models_DLinkNet import DinkNet34
from xceptionUnetPlusPlus import XceptionUnetPlusPlus

# from unet_smp import xceptionUnetPlusPlus
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomRotation
import matplotlib.pyplot as plt

class RotationWrapperModel(nn.Module):
    def __init__(self , base_model):
        super(RotationWrapperModel , self).__init__()

        self.base_model = base_model

        # Can handle batch and single images
        self.rotation_transform = RandomRotation((90,90))

    def plot_mid_data(self, x , x_90 , x_180 , x_270 , pred_0 , pred_90 , pred_180 , pred_270):

        _ , axs = plt.subplots(2,4)

        axs[0,0].imshow(x[0].cpu().permute(1,2,0))
        axs[0,1].imshow(x_90[0].cpu().permute(1,2,0))
        axs[0,2].imshow(x_180[0].cpu().permute(1,2,0))
        axs[0,3].imshow(x_270[0].cpu().permute(1,2,0))
        
        axs[1,0].imshow(pred_0[0].cpu().permute(1,2,0))
        axs[1,1].imshow(pred_90[0].cpu().permute(1,2,0))
        axs[1,2].imshow(pred_180[0].cpu().permute(1,2,0))
        axs[1,3].imshow(pred_270[0].cpu().permute(1,2,0))
        
        plt.show()

    def forward(self, x):

        pred_0 = self.base_model(x)
        if(isinstance(self.base_model, UNetBranched)):
            img, another = x
            img_90 = self.rotation_transform(img)
            another_90 = self.rotation_transform(another)
            pred_90 = self.base_model([img_90, another_90])

            img_180 = self.rotation_transform(img_90)
            another_180 = self.rotation_transform(another_90)
            pred_180 = self.base_model([img_180, another_180])

            img_270 = self.rotation_transform(img_180)
            another_270 = self.rotation_transform(another_180)
            pred_270 = self.base_model([img_270, another_270])
        elif (self.base_model.__class__.__name__ == "DeepLabV3"):
            x_90 = self.rotation_transform(x)
            pred_90 = self.base_model(x_90)

            x_180 = self.rotation_transform(x_90)
            pred_180 = self.base_model(x_180)

            x_270 = self.rotation_transform(x_180)
            pred_270 = self.base_model(x_270)
        else:
            x_90 = self.rotation_transform(x)
            pred_90 = self.base_model(x_90)

            x_180 = self.rotation_transform(x_90)
            pred_180 = self.base_model(x_180)

            x_270 = self.rotation_transform(x_180)
            pred_270 = self.base_model(x_270)

        # self.plot_mid_data(x , x_90 , x_180 , x_270 , pred_0 , pred_90 , pred_180 , pred_270)
        if (self.base_model.__class__.__name__ == "DeepLabV3"):
            # Rotating the images to align to the original images
            pred_90 = self.rotation_transform(self.rotation_transform(self.rotation_transform(pred_90['out'].argmax(1))))
            pred_180 = self.rotation_transform(self.rotation_transform(pred_180['out'].argmax(1)))
            pred_270 = self.rotation_transform(pred_270['out'].argmax(1))

            pred = torch.amax(torch.stack([pred_0['out'].argmax(1),pred_90 , pred_180 , pred_270 ]), dim=0)
        else:
            # Rotating the images to align to the original images
            pred_90 = self.rotation_transform(self.rotation_transform(self.rotation_transform(pred_90)))
            pred_180 = self.rotation_transform(self.rotation_transform(pred_180))
            pred_270 = self.rotation_transform(pred_270)

            pred = torch.amax(torch.stack([pred_0,pred_90 , pred_180 , pred_270 ]), dim=0)

        # pred = pred / 4

        return pred
    
    def load_state_dict(self, state_dict):
        return self.base_model.load_state_dict(state_dict)

    def state_dict(self):
        return self.base_model.state_dict()