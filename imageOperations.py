import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import torchvision.transforms as transforms
from submissionWriting import value_class
from sklearn.metrics import f1_score
from PIL import Image



def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("number of parameters: {}".format(num_params))

def computeF1(pred, gt, args):
    patch_pred = [img_crop(pred[i].cpu().detach().numpy(), args) for i in range(args.batch_size)]
    patch_gt = [img_crop(gt[i].cpu().detach().numpy(), args) for i in range(args.batch_size)]
    f1 = f1_score(np.array(patch_gt).ravel(), np.array(patch_pred).ravel())
    return f1


def load_image(filename):
    img = mpimg.imread(filename)
    return img


def load_images(imgDirec):
    filenames = os.listdir(imgDirec)
    n = len(filenames)
    print("Loading " + str() + " images")
    imgs = [load_image(imgDirec + filenames[i]) for i in range(n)]
    return imgs


def img_crop(im, args, w=16, h=16):
    list_labels = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            label = value_class(args, im_patch)
            list_labels.append(label)
    return list_labels

def resize_img(img, width, height):
    img = (img*255).astype("uint8")
    img = Image.fromarray(img)
    resized_img = img.resize((width,height))
    resized_img = np.asarray(resized_img)/255.0
    return resized_img
     


