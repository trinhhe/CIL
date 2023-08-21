#the code template of the exercice4 of Mathematical foundations in CV & CG adjusted to out task
from graphCut import GraphCut
import argparse
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from scipy.sparse import coo_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-preDir', '--predictionDir', required=True)
    parser.add_argument('-testDir', '--testDir', default='data/test_images')
    parser.add_argument('-l', '--lambda_val', default=0.05, type=float)

    args = parser.parse_args()

    if not os.path.exists(args.predictionDir):
        print("prediction directory does not exist!")
        exit()
    if not os.path.exists(args.testDir):
        print("original directory does not exist!")
        exit()
    names = [name for name in os.listdir(args.predictionDir) if os.path.isfile(os.path.join(args.predictionDir, name))]

    dir = args.predictionDir + 'GC_' + str(args.lambda_val)

    if not os.path.exists(dir):
        os.makedirs(dir)
              
    for name in tqdm(names):
        orig = Image.open(os.path.join(args.testDir, name))
        pred = Image.open(os.path.join(args.predictionDir, name))
        graphCut = __get_segmented_image(orig, pred, args.lambda_val)
        path = os.path.join(dir, name)
        Image.fromarray(graphCut).save(path)



def __get_segmented_image(image, label, lambda_param):
    image_array = np.asarray(image)
    pred_mask = np.asarray(label)
    height, width = np.shape(image_array)[:2]
    r = np.random.rand(height, width)
    roadPercentage = np.sum(pred_mask > 0) / (height * width)
    mask_1 = r > roadPercentage 
    mask_0 = r < roadPercentage 
    ones = pred_mask > 0
    seed_fg = np.stack((np.where(ones * mask_1))).T
    seed_bg = np.stack((np.where((1-ones) * mask_0))).T

    # TASK 2.1 : get the color histogram for the unaries
    hist_res = 32
    cost_fg = __get_color_histogram(image_array, seed_fg, hist_res) + 1e-10
    cost_bg = __get_color_histogram(image_array, seed_bg, hist_res) + 1e-10

    # TASK 2.2-2.3 : set the unaries and the pairwise terms
    unaries = __get_unaries(image_array, lambda_param, cost_fg, cost_bg, seed_fg, seed_bg)
    adj_mat = __get_pairwise(image_array, sigma=5)

    # TODO: TASK 2.4 : perform graph cut
    g = GraphCut(height * width, adj_mat.count_nonzero() * 2)
    g.set_unary(unaries)
    g.set_neighbors(adj_mat)
    g.minimize()
    labels = g.get_labeling()
    labels = np.reshape(labels, (height, width))
    img_gc = labels * 255
    return img_gc.astype(np.uint8)


def __get_color_histogram(image, seed, hist_res):
    """
        Compute a color histograms based on selected points from an image
        
        :param image: color image
        :param seed: Nx2 matrix containing the the position of pixels which will be
                    used to compute the color histogram
        :param histRes: resolution of the histogram
        :return: hist: color histogram
    """
    pixels = image[seed[:,0], seed[:,1]]
    hist_step = 256.0 / 32.0
    indexes_rgb = (pixels / hist_step).astype(int)
    histogram = np.zeros((hist_res, hist_res, hist_res))
    for ind in indexes_rgb:
        #print("ind: ", ind)
        #print("ele: ", ind[0])
        histogram[ind[0],ind[1],ind[2]] += 1

    hist = ndimage.gaussian_filter(histogram, 0.5)
    normalized_hist = hist / np.sum(hist)

    return normalized_hist

#the weights connecting each pixel node to the source anf the sink
# unless the col row belong to the foreground seed or background seed we assign R_p andd R_obj= lambda * -lnPr(Ip(Ip|O))
def __get_unaries(image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):
    """
        :param image: color image as a numpy array
        :param lambda_param: lamdba as set by the user
        :param hist_fg: foreground color histogram
        :param hist_bg: background color histogram
        :param seed_fg: pixels marked as foreground by the user
        :param seed_bg: pixels marked as background by the user
        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
    """
    hist_res = hist_fg.shape[0]
    hist_step = 256.0 / 32.0
    indexes_rgb = (image / hist_step).astype(int)
    #print("indexes_rgb: ", indexes_rgb)
    unaries = np.empty((image.shape[0], image.shape[1], 2))


    #values taken from the table in https://www.csd.uwo.ca/~yboykov/Papers/iccv01.pdf
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
                ind = indexes_rgb[i, j]
                cost_fg =  -np.log(hist_fg[ind[0], ind[1], ind[2]])
                cost_bg =  -np.log(hist_bg[ind[0], ind[1], ind[2]])
                unaries[i, j] = [lambda_param * cost_bg, lambda_param * cost_fg]
    for i, j in seed_fg:
        unaries[i, j] = [np.inf, 0]
    for i, j in seed_bg:
        unaries[i, j] = [0, np.inf]

    newUnaries = unaries.reshape((image.shape[0]*image.shape[1], 2))
    return newUnaries

#the weights between each 2 neighboring pixels: B() = exp((I_p - I_q) / 2 * sig^2)^2 / dist(p,q)
def __get_pairwise(image, sigma=5):

    height, width = image.shape[:2]

    intensity = np.sum(np.square(image[:-1,:] - image[1:,:]), axis=2)
    intensity = np.exp(-(intensity / (2 * sigma**2)))
    dataVer = intensity.reshape(-1)
    #############################################################
    intensity = np.sum(np.square(image[:,:-1] - image[:,1:]), axis=2)
    intensity = np.exp(-(intensity / (2 * sigma**2)))
    dataHor = intensity.reshape(-1)
    #############################################################
    intensity = np.sum(np.square(image[:-1,:-1] - image[1:,1:]), axis=2)
    dist_scale = 1 / np.sqrt(2)
    intensity = np.exp(-(intensity / (2 * sigma**2))) * dist_scale
    dataDiag = intensity.reshape(-1)
    #############################################################
    intensity_ssd = np.sum(np.square(image[1:,:-1] - image[:-1,1:]), axis=2)
    intensity = np.exp(-(intensity_ssd / (2 * sigma**2))) * dist_scale
    dataDiagOth = intensity.reshape(-1)

    data = np.hstack((dataHor, dataVer, dataDiag, dataDiagOth))
    #############################################################
    rowsVer = np.arange(width*(height-1))
    rowsHor = np.arange(1,width*height+1).reshape(height,width)[:,:-1].reshape(-1) - 1
    rowsDiag = rowsHor[:-width+1]
    rowsDiagOth = rowsHor[:-width+1] + 1
    rows = np.hstack((rowsHor, rowsVer, rowsDiag, rowsDiagOth))
    #############################################################
    colsVer = np.arange(width*(height-1)) + width
    colsHor = np.arange(1,width*height+1).reshape(height,width)[:,:-1].reshape(-1)
    temp = np.arange(1,width*height+1).reshape(height,width)[:,:-1].reshape(-1) - 1
    colsDiag = temp[:-width+1] + width + 1
    colsDiagOth = temp[:-width+1] + width 
    cols = np.hstack((colsHor, colsVer, colsDiag, colsDiagOth))
    #############################################################
    pixelSize = height*width
    pairwise_mat = coo_matrix((data, (rows, cols)), shape=(pixelSize , pixelSize))

    return pairwise_mat

if __name__ == "__main__":
    main()
