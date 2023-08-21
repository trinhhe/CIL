import numpy as np
import argparse
import os
from PIL import Image
import torchvision as tv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDir', action='append', required=True)
    parser.add_argument('-o', '--outputDir', default='ensembleOutput')
    args = parser.parse_args()

    dirNums = len(args.inputDir)
    if dirNums < 2:
        print('take more than 2 directories')
        exit()

    for eachDir in args.inputDir:
        if not os.path.exists(eachDir):
            print('invalid directory: ', eachDir)
            exit()

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    listNames = [imgName for imgName in os.listdir(args.inputDir[0])]
    maj = (dirNums + 2) // 2
    for imgName in listNames:
        mask = np.zeros((608, 608), dtype=np.int)
        for outModelDir in args.inputDir:
                s = (Image.open(os.path.join(outModelDir, imgName)))
                #s = Image.fromarray(s)
                s = np.array(s)
                if(s.shape == (608, 608, 3)): 
                    news = s[:,:,0]
                    imgize = tv.transforms.ToPILImage()
                    news = imgize(news)
                    news.save(os.path.join(outModelDir, imgName))
                mask += Image.open(os.path.join(outModelDir, imgName))
        mask = np.where(mask >= maj * 255, np.ones_like(mask), np.zeros_like(mask))
        mask = (mask * 255).astype(np.uint8)
        path = os.path.join(args.outputDir, imgName)
        Image.fromarray(mask).save(path)



if __name__ == "__main__":
    main()