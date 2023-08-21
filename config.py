from os import listdir
from os.path import isfile, join

base_DIR = "data"

base_DIR_training = join("data","training")
base_DIR_testing = join("data","test_images")
base_DIR_aug = join("data","training_augmented")

ground_truth_PATH_TRAIN = join(base_DIR_training , "groundtruth") 
images_PATH_TRAIN = join(base_DIR_training , "images")

ground_truth_PATH_AUG = join(base_DIR_aug , "groundtruth") 
images_PATH_AUG = join(base_DIR_aug, "images")

ROTATE = True
TRANSLATE = True
ADD_NOISE = True
BLUR = True
MIRROR = True
BRIGHTNESS = False
CONTRAST = False
PCA = False
