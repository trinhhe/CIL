import argparse
import os
import numpy as np
import torch
import torchvision as tv
from tqdm import tqdm
from helper import cross_entropy, ImageLoader, getModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-trainDir', '--trainDir', default='data/trainingCr/images')
    parser.add_argument('-trainGT_DIR', '--trainGT_DIR', default='data/trainingCr/groundtruth')
    parser.add_argument('-testDir', '--testDir', default='data/test_images')
    parser.add_argument('-epoch', '--nepochs', type=int, default=60)
    parser.add_argument('-lr', '--learningRate', type=float, default=0.0001)
    parser.add_argument('-m', '--model', default='mobile') 
    parser.add_argument('-bs', '--batchSize', type=int, default=4)
    args = parser.parse_args()

    testData = ImageLoader(testDir=args.testDir)
    trainingData = ImageLoader(imgDirTrain=args.trainDir, gtDirTrain=args.trainGT_DIR)

    args.outputDir =  args.model + '_PRED'
    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)

    model = getModel(args.model)
    train(args, model, trainingData)
    test(args, model, testData)


def test(args, ml, dataSet):
    data = torch.utils.data.DataLoader(dataSet, batch_size=args.batchSize)
    ids = np.asarray([os.path.join(args.testDir, filename)for filename in os.listdir(args.testDir)])
    i = 0
    with torch.no_grad():
        ml.eval()
        for batch in tqdm(data):
            images = batch['image'].cuda()
            outs = ml(images)['out']
            preds = outs.argmax(1)
            for label in preds:
                label = torch.where(label == 1, torch.ones_like(label) * 255, torch.zeros_like(label)).byte()
                imageOf = tv.transforms.ToPILImage()
                label = imageOf(label.cpu())
                name = ids[i]
                i+=1
                label.save(os.path.join(args.outputDir, os.path.basename(name)))

def train(args, ml, dataSet):
    data = torch.utils.data.DataLoader(dataSet, batch_size=args.batchSize, shuffle=True)
    optimizer = torch.optim.Adam(ml.parameters(), lr = args.learningRate, weight_decay=0.0)
    ml.train()
    for e in tqdm(range(args.nepochs)):
        for batch in tqdm(data):
            x = batch['image'].cuda()
            pred = ml(x)['out']
            y = batch['gt'].cuda()
            loss = cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm.write('loss:{}'.format(loss.item()))
   


if __name__ == "__main__":
    main()
