#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import sys

thres = 0.25 

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > thres:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    img_number = int(re.search(r"\d+", image_filename.split("/")[-1]).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *filenames):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':
    filenames = []
    outModelDir = sys.argv[1]
    submission_filename = os.path.join(outModelDir, 'submission.csv')
    files = [f for f in os.listdir(outModelDir) if "png" in f]
    for f in files:
        filename = os.path.join(outModelDir, f)
        print("writing ", filename, "...")
        filenames.append(filename)
    masks_to_submission(submission_filename, *filenames)
