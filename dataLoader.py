from operator import index
from PIL import Image , ImageFilter , ImageEnhance
from torch.utils import data
from imageOperations import *
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage.draw import line
from matplotlib import cm
from matplotlib import image as matimg
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.transform import resize

class ImageLoader(data.Dataset):
    def __init__(self, args, mode = "train" , resize = False, width = None, height = None ):

        # initialize parameters
        self.mode = mode
        self.resize = resize
        self.height = height
        self.width = width

        self.train_path = args.train_path
        self.gt_path = args.gt_path
        self.test_path = args.test_path

        # list of all files of a directory
        self.trainFiles = os.listdir(args.train_path)
        self.gtFiles = os.listdir(args.gt_path)
        self.testFiles = os.listdir(args.test_path)

        self.ids = [int(filename.split('_')[1].split('.')[0]) for filename in os.listdir(args.test_path)]

        # dataloader
        print('********************%s dataloader***********************'%mode)
        if mode == 'train':
            print("found " + str(len( self.trainFiles )) + " images")
        elif mode == "test":
            print("found " + str(len( self.testFiles )) + " images")

    def __getitem__(self, index):
        if self.mode == 'train':
            img = load_image(self.train_path + self.trainFiles[index])
            gt = load_image(self.gt_path + self.trainFiles[index])

            if self.resize:
                img = resize_img(img ,self.width, self.height)
                gt = resize_img(gt , self.width, self.height)

            return img, gt

        elif self.mode == 'test':
            img = load_image(self.test_path + 'test_' + str(self.ids[index]) + ".png")
            return img

    def __len__(self):
        if self.mode == 'train':
            return len(self.trainFiles)
        elif self.mode == 'test':
            return len(self.testFiles)

class ImageLoaderAndAnother(data.Dataset):
    def __init__(self, args, mode = "train" , resize = False, width = None, height = None ):

        # initialize parameters
        self.mode = mode
        self.resize = resize
        self.height = height
        self.width = width

        self.secondary_image = args.secondary_image
        self.train_path = args.train_path
        self.gt_path = args.gt_path
        self.test_path = args.test_path
        self.train_mask_path = args.train_mask_path
        self.test_mask_path = args.test_mask_path

        # list of all files of a directory
        self.trainFiles = os.listdir(args.train_path)
        self.gtFiles = os.listdir(args.gt_path)
        self.testFiles = os.listdir(args.test_path)

        self.ids = [int(filename.split('_')[1].split('.')[0]) for filename in os.listdir(args.test_path)]

        # dataloader
        print('********************%s dataloader***********************'%mode)
        if mode == 'train':
            print("found " + str(len( self.trainFiles )) + " images")
        elif mode == "test":
            print("found " + str(len( self.testFiles )) + " images")

    def calc_another(self,index):

        if self.mode == "train":
            another = Image.open(self.train_mask_path + self.trainFiles[index])
        else:
            another = Image.open(self.test_mask_path + 'test_' + str(self.ids[index]) + ".png")
        
        another = np.asarray(another.convert("RGB"))*255
        if self.resize:
            another = resize_img(another , self.width , self.height)

        return another

    def hough_calc_another(self, img):

        # Using the regular input image
        img_RGB = (img*255).astype(np.uint8)[:, :, 1]

        # Making a fake black background
        black_back = np.zeros((img_RGB.shape[0], img_RGB.shape[1]))

        # Show the image
        imgShow = Image.fromarray(img_RGB)
        imgShow = imgShow.filter(ImageFilter.BoxBlur(6))
        image = np.array(imgShow)

        # Start drawing on a canvas
        fig = Figure()
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)

        canvas = FigureCanvas(fig)
        ax = fig.gca()

        # Set a precision of certain degrees
        tested_angles = np.linspace(np.pi / 2, -np.pi / 2, 2, endpoint=False)

        # Classic straight-line Hough transform

        # Probabilistic Hough transform
        edges = canny(image, 2, 1, 25)
        # Detects all angles of lines
        lines = probabilistic_hough_line(edges, line_length=20,
                                        line_gap=3)

        ax.imshow(black_back, cmap=cm.gray)

        ax.set_axis_off()
        ax.axis('tight')
        
        # Generating figure for Probabilistic Hough transform
        for line in lines:
            p0, p1 = line
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]), 'w-')
        
        canvas.draw()       # draw the canvas, cache the renderer

        buf = canvas.buffer_rgba()
        # ... convert to a NumPy array ...
        X = np.asarray(buf)
        # ... and pass it to PIL.
        im = Image.fromarray(X)
        im = im.resize((img_RGB.shape[0], img_RGB.shape[1]))
        im = np.array(im)[:, :, 0:3]

        another = im
        return another

    def __getitem__(self, index):
        if self.mode == 'train':
            img = load_image(self.train_path + self.trainFiles[index])
            gt = load_image(self.gt_path + self.trainFiles[index])

            if self.resize:
                img = resize_img(img ,self.width, self.height)
                gt = resize_img(gt , self.width, self.height)

            if self.secondary_image == "graphCut":
                another = self.calc_another(index)
            else:
                another = self.hough_calc_another(img)
            
            
            return (img , another), gt

        elif self.mode == 'test':
            img = load_image(self.test_path + 'test_' + str(self.ids[index]) + ".png")
            if self.secondary_image == "graphCut":
                another = self.calc_another(index)
            else:
                another = self.hough_calc_another(img)
            
            return (img , another)

    def __len__(self):
        if self.mode == 'train':
            return len(self.trainFiles)
        elif self.mode == 'test':
            return len(self.testFiles)

class ImageLoader3Channel(data.Dataset):
    def __init__(self, args, mode = "train" , resize = False, width = None, height = None ):

        # initialize parameters
        self.mode = mode
        self.resize = resize
        self.height = height
        self.width = width

        self.train_path = args.train_path
        self.gt_path = args.gt_path
        self.test_path = args.test_path

        # list of all files of a directory
        self.trainFiles = os.listdir(args.train_path)
        self.gtFiles = os.listdir(args.gt_path)
        self.testFiles = os.listdir(args.test_path)

        self.ids = [int(filename.split('_')[1].split('.')[0]) for filename in os.listdir(args.test_path)]

        # dataloader
        print('********************%s dataloader***********************'%mode)
        if mode == 'train':
            print("found " + str(len( self.trainFiles )) + " images")
        elif mode == "test":
            print("found " + str(len( self.testFiles )) + " images")

    def hough_calc_another(self, img):
        # Using a satelite image instead

        # Using the regular input image
        img_RGB = (img*255).astype(np.uint8)[:, :, 1]


        # Making a fake black background
        black_back = np.zeros((img_RGB.shape[0], img_RGB.shape[1]))

        # Show the image
        imgShow = Image.fromarray(img_RGB)
        # imgShow.show()
        imgShow = imgShow.filter(ImageFilter.BoxBlur(6))
        image = np.array(imgShow)

        # Start drawing on a canvas
        fig = Figure()
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)

        canvas = FigureCanvas(fig)
        ax = fig.gca()

        # Set a precision of certain degrees
        tested_angles = np.linspace(np.pi / 2, -np.pi / 2, 2, endpoint=False)

        # Probabilistic Hough transform
        edges = canny(image, 2, 1, 25)
        # Detects all angles of lines
        lines = probabilistic_hough_line(edges, line_length=20,
                                        line_gap=3)

        ax.imshow(black_back, cmap=cm.gray)

        ax.set_axis_off()
        ax.axis('tight')

        # Generating figure for Probabilistic Hough transform
        for line in lines:
            p0, p1 = line
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]), 'w-')
        
        canvas.draw()       # draw the canvas, cache the renderer

        buf = canvas.buffer_rgba()
        # ... convert to a NumPy array ...
        X = np.asarray(buf)
        # ... and pass it to PIL.
        im = Image.fromarray(X)
        im = im.resize((img_RGB.shape[0], img_RGB.shape[1]))
        im = np.array(im)[:, :, 0:3]

        another = im
        return another

    def calc_another(self,img):
        another = Image.fromarray((img*255).astype(np.uint8))
        another = another.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
        another = np.asarray(another).copy()

        return another

    def __getitem__(self, index):
        if self.mode == 'train':
            img = load_image(self.train_path + self.trainFiles[index])
            gt = load_image(self.gt_path + self.trainFiles[index])

            if self.resize:
                img = resize_img(img ,self.width, self.height)

                # Do some image magic first to get grayscale
                imgg = (img * 255).astype(np.uint8)
                imgg = Image.fromarray(imgg).convert("L")

                hough_channel = self.hough_calc_another(img)[:, :, 0]
                img_channel = np.asarray(imgg)
                canny_channel = self.calc_another(img)[:, :, 0]

                img_hough_canny = np.stack((img_channel, hough_channel, canny_channel), axis=2)
                
                # Stacked together as img grayscale, hough, and canny
                img = img_hough_canny
                gt = resize_img(gt , self.width, self.height)

            return img, gt

        elif self.mode == 'test':
            img = load_image(self.test_path + 'test_' + str(self.ids[index]) + ".png")
            return img

    def __len__(self):
        if self.mode == 'train':
            return len(self.trainFiles)
        elif self.mode == 'test':
            return len(self.testFiles)