# Computational Intelligence Lab - 2021

### Team Jedi
### Authors: Elham Amin Mansour, Dhruv Agrawal, Henry Trinh, Jiaqi Chen
17-342-114, 20-946-059, 15-937-816, 20-947-537


  This paper focuses on improving the state of the art segmentation methods, specifically for satellite road images. As the current SOA in segmentation is deep learning based, our contribution is in exploring and incorporating novel ideas and tweaks by making "structural changes" related to model architecture and "data changes" related to data pre- and post-processing. We propose the model Branched UNet that further is extended to a GAN model and beats our state of the art baseline scores of 0.85148 from UNet and 0.88713 from DeepLab101. Our final model achieves an accuracy score of 0.91334 on Kaggle.



baseline1: Vanilla Unet
baseline2:  DeepLabV3-ResNet101
## Setting up the environment
Execute the following commands to setup the execution environment

1. `pip install virtualenv` 

2. `virtualenv cil-team-jedi`
   
3. `cil-team-jedi/Scripts/activate`
   
4. `./env_setup.sh`

## Croppping each training image into 5 256x256 images
1. 
    `mkdir ./data/trainingCr/images`
2.  
    `mkdir ./data/trainingCr/groundtruth`
3.
    `python3 cropping.py`

## Cropping any 400x400 images into 5 256x256 images
`python3 cropping.py --images_path  ./path_to_image_folder --save_path ./path_to_save_cropped_images`
## Training using main.py

1. To start new training using model UNet for n epochs and saving models to save/UNet

    `python main.py --model UNet --epochs 50 --model_save_dir save/UNet`

2. To load a UNet model from save/UNet and create the submissions file in output.

    `python main.py --model UNet --only_pred True --model_path save/UNet/model_99 --outputPath output`

3. To load a UNet model and visualize 10 predictions on the test dataset

    `python main.py --model UNet --visualize True --visualize_samples 10 --visualize_dataset test --model_path save/Unet/model_99`

4. To  use 3 channel Imageloader (Grayscale , Canny and Hough) for training UNet

    `python main.py --model UNet --epochs 50 --model_save_dir save/UNet --image_loader ImageLoader3Channel`

5. To resume training on a pretrained UNet model 

    `python main.py --model UNet --epochs 30 --model_save_dir save/UNet --start_epoch 50 --model_path save/UNet/model_49`

6. To save the predicted images to a folder

    `python main.py --model UNet --visualize True --save True --save_dir data/prediction/UNet --model_path save/UNet/model_49`

7. To use RotationWrapper with a trained UNet model:
   
   `python main.py --model UNet --rotate_wrapper True --visualize True --save True --save_dir data/prediction/UNet_Rotated --model_path save/UNet/model_49`

## Training using main_with_another.py (UNet Branched)

Similar to main.py. Specifically for training using UNet_branched

1. To start new training using model UNet for n epochs and saving models to save/UNet_branched with Hough Transform as secondary image input

    `python main_with_another.py --model UNetBranched --epochs 50 --model_save_dir save/UNet_branched --secondary_image hough`

2.  To start new training using model UNet for n epochs and saving models to save/UNet_branched with graph cut as secondary image input

    `python main_with_another.py --model UNetBranched --epochs 50 --model_save_dir save/UNet_branched --secondary_image graphCut`

3. To load a UNet model from save/UNet_branched and create the submissions file in output.

    `python main_with_another.py --model UNetBranched --only_pred True --model_path save/UNet_branched/model_99 --outputPath output`

4. To load a UNet model and visualize 10 predictions on the test dataset

    `python main_with_another.py --model UNetBranched --visualize True --visualize_samples 10 --visualize_dataset test --model_path save/Unet_branched/model_99`

5. To resume training on a pretrained UNetBranched model 

    `python main_with_another.py --model UNetBranched --epochs 30 --model_save_dir save/UNet_branched --start_epoch 50 --model_path save/UNet_branched/model_49`

6. To save the predicted images to a folder

    `python main.py --model UNetBranched --visualize True --save True --save_dir data/prediction/UNet_branched --model_path save/UNet_branched/model_49`

Does not have `--image_loader` option since it is supposed to only use `ImageLoaderWithAnother` data loader.


## Training using modelRun

1. For training and visualizing results

`python3 modelRun.py -m deeplab50`

`python3 modelRun.py -m deeplab101`

`python3 modelRun.py -m mobile`

2. For making the submission file:

`python3 mask_to_submission.py ./mobile_PRED`


## Training using main_GAN.py

1. To start new training for GAN model (Generator: XceptionUnet++ or UNetBranched, Discriminator: PatchGAN) for n epochs, m batch size and saving models to save/GAN

    `python main_GAN.py --G XceptionUnetPlusPlus --D NLayerDiscriminator --epochs 50 --batch_size 4 --model_save_dir save/XceptionUnetPlusPlus_GAN`

    `python main_GAN.py --G UNetBranched --image_loader ImageLoaderAndAnother --model_save_dir save/UNetBranched_GAN --train_mask_path data/prediction/groundtruth_256_GC_0.1/ --test_mask_path data/prediction/deeplab50_croppedGC_0.1/ --epochs 100 --batch_size 4`

2. To load a generator from save/GAN and create the submissions file in output

    `python main_GAN.py --G XceptionUnetPlusPlus --only_pred True --G_path save/XceptionUnetPlusPlus_GAN/G_49`

    `python main_GAN.py --G UNetBranched --image_loader ImageLoaderAndAnother --only_pred True --G_path save/UNetBranched_GAN/G_99`
3. To save the predicted images to a folder

    `python main_GAN.py --G XceptionUnetPlusPlus --visualize True --save True --G_path save/XceptionUnetPlusPlus_GAN/G_99 --save_dir data/prediction/XceptionUnetPlusPlus_GAN`

    `python main_GAN.py --G UNetBranched --image_loader ImageLoaderAndAnother --visualize True --G_path save/UNetBranched_GAN/G_99 --save_dir data/prediction/UNetBranched_GAN`

## Training using modelRunGAN.py

1. Start training GAN model with resnet50, deeplab50, deeplab101 or mobile

    `python modelRunGAN.py --G deeplab101 --model_save_dir save/deeplab101_GAN --epochs 100 --batch_size 4`

2. Load model from save/deeplab101_GAN and create submissions file in output

    `python modelRunGAN.py --G deeplab101 --only_pred True --G_path save/deeplab101_GAN/G_99`

3. Visualize predictions on test set and save in output/predictions folder

    `python modelRunGAN.py --G deeplab101 --visualize True --G_path save/deeplab101_GAN/G_99 --save_dir output/prediction/deeplab101_GAN`
## For postprocesssing the images predicated from a model with graphcut

`python3 graphCutRun.py --predictionDir ./deeplab50_cropped --testDir ./data/test_images  --lambda_val 0.1`

## For ensembling multiple folders
[google drive link containing predictions of different models](https://drive.google.com/drive/folders/1HLwVMhrQdA-HYwajOlrHbLafYAzGyfSL?usp=sharing)

`pathi = path to i-th prediction images`

For our best ensemble, we used the following models:

`path1="./prediction/deeplab50_PRED_Cr_noAugment"`

`path2="./prediction/deeplab101_PRED_Cr_noAugment"`

`path3="./prediction/mobile_PRED_Cr_noAugment"`

`path4="./prediction/UNet_Denoised"`

`path5="./prediction/UNet_Denoised_1channel"`

`path7 ="./prediction/UNet_vanilla"`

`path8 ="./prediction/UNet_Branched_GC_Original"`

`path9 ="./prediction/DinkNet34_59_cropped"`


`python3 ensemble.py  -i $path1 -i $path2 -i $path3 -i $path4 -i $path5 -i $path6 -i $path7 -i $path8 -i $path9`

## Commands for best submission
Make sure that data/trainingCr/images, data/trainingCr/groundtruth and data/prediction folders don't exist.

1. 
    `mkdir ./data/trainingCr/images && mkdir ./data/trainingCr/groundtruth && python cropping.py && python graphCutRun.py --predictionDir ./data/trainingCr/groundtruth/ --testDir ./data/trainingCr/images/ -l 0.1 && mkdir ./data/prediction && mv ./data/trainingCr/groundtruth/GC_0.1 ./data/prediction/ && mv ./data/prediction/GC_0.1 ./data/prediction/groundtruth_256_GC_0.1`

2.
    `python main_GAN.py --epochs 50 --batch_size 4 --G XceptionUnetPlusPlus --model_save_dir save/XceptionUnetPlusPlus_GAN`

3.
    `python main_GAN.py --G XceptionUnetPlusPlus --G_path save/XceptionUnetPlusPlus_GAN/G_49 --rotate_wrapper True --save True --visualize True --save_dir data/prediction/XceptionUnetPlusPlus_GAN_rotate`
    
4.
    `python graphCutRun.py --l 0.1 --predictionDir data/prediction/XceptionUnetPlusPlus_GAN_rotate`

5.
    `python main_GAN.py --epochs 100 --batch_size 4 --G UNetBranched --image_loader ImageLoaderAndAnother --loss L1Loss --model_save_dir save/UNetBranched_GAN --train_mask_path data/prediction/groundtruth_256_GC_0.1/ --test_mask_path data/prediction/XceptionUnetPlusPlus_GAN_rotateGC_0.1/`

6. 
    `python main_GAN.py --G UNetBranched --rotate_wrapper True --image_loader ImageLoaderAndAnother --only_pred True --G_path save/UNetBranched_GAN/G_99 --test_mask_path data/prediction/XceptionUnetPlusPlus_GAN_rotateGC_0.1/`

    submission.csv file is in output folder
    (note: this command sequence will take about 8h on a Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz and Nvidia Geforce GTX 1070)
