import numpy as np
import os, sys
import shutil
import matplotlib.image as mpimg
from skimage import io
import argparse

# helper
def load(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def png_dataset_save(dataset, output_path):
    for i in range(len(dataset)):
        filename='/' + str(i) + '.png'
        io.imsave(output_path + filename, dataset[i])
        
# takes a 400 by 400 image as input and returns 5 256x256 images as output
def crop_400_to_256(img):
    top_left = img[0:256, 0:256]
    top_right = img[0:256, 144:400]
    bottom_left = img[144:400, 0:256]
    bottom_right = img[144:400, 144:400]
    center = img[80:336, 80:336]
    
    return [top_left, top_right, bottom_left, bottom_right, center]

# takes a 608x608 image for the input and gives 9 256x256 images as output
def crop_608_to_256(img):
    interval1 = slice(0,256) 
    interval2 = slice(176,432) 
    interval3 = slice(352,608) 
    
    # 9 patches
    topRight = img[interval1, interval3]
    topCenter = img[interval1, interval2]
    topLeft = img[interval1, interval1]
    centerRight = img[interval2, interval3]
    centerCenter = img[interval2, interval2]
    centerLeft = img[interval2, interval1]
    bottomRight = img[interval3, interval3]
    bottomCenter = img[interval3, interval2]
    bottomLeft = img[interval3, interval1]

    
    return [topLeft, topCenter, topRight, centerLeft, centerCenter, centerRight, bottomLeft, bottomCenter, bottomRight]

def uncrop_256_to_608(imgs):
    interval1= slice(0,256) 
    interval2 = slice(176,432)
    interval3 = slice(352,608)

    outShape = (608,608)

    topRight = np.zeros(outShape)
    topCenter = np.zeros(outShape)
    topLeft = np.zeros(outShape)
    centerRight = np.zeros(outShape)
    centerCenter = np.zeros(outShape)
    centerLeft = np.zeros(outShape)
    bottomRight = np.zeros(outShape)
    bottomCenter = np.zeros(outShape)
    bottomLeft = np.zeros(outShape)
    
    
    topLeft[interval1, interval1] = imgs[0]
    topCenter[interval1, interval2] = imgs[1]
    topRight[interval1, interval3] = imgs[2]
    centerLeft[interval2, interval1] = imgs[3]
    centerCenter[interval2, interval2] = imgs[4]
    centerRight[interval2, interval3] = imgs[5]
    bottomLeft[interval3, interval1] = imgs[6]
    bottomCenter[interval3, interval2] = imgs[7]
    bottomRight[interval3, interval3] = imgs[8]

    output = np.max([topRight, topCenter, topLeft, centerRight, centerCenter, centerLeft, bottomRight, bottomCenter, bottomLeft], axis=0).astype(np.uint8)
    return output


def cropData(save):
    rootDir  = "./data/training/"

    imageDir = rootDir  + "images/"
    try:
        files = os.listdir(imageDir)
    except:
        raise SystemExit
        
    n = min(100, len(files)) 
    imgs = img_float_to_uint8([load(imageDir + files[i]) for i in range(n)])
        
    imageDir = rootDir  + "groundtruth/"
    files = os.listdir(imageDir)
    gts = img_float_to_uint8([load(imageDir + files[i]) for i in range(n)])

    ########################################

    trainShape = list(imgs.shape)
    trainCropShape = (5*trainShape[0], 256, 256, 3)
    trainCrop = np.empty(trainCropShape).astype(np.uint8)

    for index in range(len(imgs)):
        image = imgs[index]
        trainCrop[5*index:5*index+5] = np.asarray(crop_400_to_256(image))

    gtShape = list(gts.shape)
    gtCropShape = (5*gtShape[0], 256, 256)
    gtCrop  = np.empty(gtCropShape).astype(np.uint8)

    for index in range(len(gts)):
        image = gts[index]
        gtCrop[5*index:5*index+5] = np.asarray(crop_400_to_256(image))

    #######################################

    if save:
        png_dataset_save(trainCrop, "./data/trainingCr/images")
        png_dataset_save(gtCrop , "./data/trainingCr/groundtruth")
    
    return trainCrop, gtCrop 

def cropData1(args):
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    try:    
        files = os.listdir(args.images_path)
    except:
        raise SystemExit

    n = len(files)
    imgs = img_float_to_uint8([load(args.images_path + files[i]) for i in range(n)])
    Shape = list(imgs.shape)
    CropShape = (5*Shape[0], 256, 256)
    Crop = np.empty(CropShape).astype(np.uint8)
    for index in range(len(imgs)):
        image = imgs[index]
        Crop[5*index:5*index+5] = np.asarray(crop_400_to_256(image))
    
    png_dataset_save(Crop, args.save_path)

    return Crop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--images_path", type=str , default=None, help="Path where images are to crop")
    parser.add_argument("--save_path", type=str , default=None, help="Path where cropped image are to be saved")
    args = parser.parse_args()
    #crop original training data (img and gt)
    if(args.images_path is None):
        cropData(True)
    #crop some images of 400x400 to 256x256 from images_path to save_path
    else:
        cropData1(args)