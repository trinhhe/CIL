import os
import numpy as np
import torch
import matplotlib.image as mpimg



# label for each patch
def value_class(args, patch):
    df = np.mean(patch)
    if df > args.threshold:
        return 1
    else:
        return 0


def oneResultSubmission(im, args, index):
    patchSize = 16
    for j in range(0, im.shape[1], patchSize):
        for i in range(0, im.shape[0], patchSize):
            patch = im[i:i + patchSize, j:j + patchSize]
            label = value_class(args, patch)
            yield("{:03d}_{}_{},{}".format(index, j, i, label))


def resultSubmission(filename, images, args, ids):
    with open(filename, 'w') as f:
        f.write('id,prediction\n')
        for i, im in enumerate(images):
            f.writelines('{}\n'.format(s) for s in oneResultSubmission(im, args, ids[i]))

