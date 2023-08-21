from typing import List
from sklearn.utils import tosequence
from unet import *
from models_DLinkNet import *
from xceptionUnetPlusPlus import *
from Discriminator import *
from RotationWrapperModel import *
from PIL import Image
import matplotlib.pyplot as plt
from imageOperations import *
import argparse
from dataLoader import ImageLoader, ImageLoaderAndAnother, ImageLoader3Channel
from torch.nn import L1Loss
import torch.optim as optim
from losses import *
from submissionWriting import *
from torch.utils import data
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from time import time
from tqdm import tqdm

def run(args ,G, D, start_epoch = 0 , device = "cpu"):

    # build dataset
    trainSet = eval(args.image_loader)(args, mode='train', resize=True , height=256 , width=256)
    
    # build data loader
    trainLoader = torch.utils.data.DataLoader(dataset = trainSet, batch_size = args.batch_size, shuffle=True)
    
    # build G, D and train
    print(G)
    print(D)

    criterion = eval(args.loss)()
    criterionGAN = eval(args.GANloss)()

    optimizer_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))

    lossesG = []
    lossesD = []
    F1s = []

    schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, factor=args.gamma, patience=5, threshold=0.0001, eps=1e-06, verbose=True)
    schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, factor=args.gamma, patience=5, threshold=0.0001, eps=1e-06, verbose=True)
    maxF1 = 0

    batches = len(trainLoader)
    G.train()
    D.train()

    t0 = time()
    # start training
    for epoch in tqdm(range(args.epochs)):
        avgCostG = 0
        avgCostD = 0
        avgF1 = 0
        with tqdm(total=batches) as pbar:
            for _ , (x, gt) in enumerate(trainLoader):
                if(args.G == "UNetBranched"):
                    img, another = x
                    img = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
                    another = another.permute(0,3,1,2).type(torch.FloatTensor).to(device)
                    x = [img , another]
                else:
                    x = x.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
                
                gt = torch.round(gt)
                gt = gt.type(torch.FloatTensor).to(device)
                fake_gt = G(x)
                # update discriminator
                optimizer_D.zero_grad()
                # Fake, note: detach to stop backprop to generator
                #conditional GAN; we feed both input and output to the discriminator
                if(args.G == "UNetBranched"):
                    fake_img_gt = torch.cat((x[0], fake_gt), 1) 
                else:
                    fake_img_gt = torch.cat((x, fake_gt), 1) 

                pred_fake = D(fake_img_gt.detach())
                loss_D_fake = criterionGAN(pred_fake, torch.zeros(pred_fake.size()).cuda())
                # Real
                if(args.G == "UNetBranched"):
                    real_img_gt = torch.cat((x[0], gt.unsqueeze(1)), 1) 
                else:
                    real_img_gt = torch.cat((x, gt.unsqueeze(1)), 1)
                
                pred_real = D(real_img_gt)
                loss_D_real = criterionGAN(pred_real, torch.ones(pred_real.size()).cuda())

                loss_D = (loss_D_fake + loss_D_real) * 0.5
                loss_D.backward()
                optimizer_D.step()

                # update generator
                optimizer_G.zero_grad()
                # generator fakes discriminator
                if(args.G == "UNetBranched"):
                    fake_img_gt = torch.cat((x[0], fake_gt), 1) 
                else:
                    fake_img_gt = torch.cat((x, fake_gt), 1)
                
                pred_fake = D(fake_img_gt)
                loss_G_GAN = criterionGAN(pred_fake, torch.ones(pred_fake.size()).cuda())
                # generator also optimizes on loss funciton e.g. IoULoss
                loss_G_loss = criterion(fake_gt.squeeze(1), gt) * args.lambda_loss
                loss_G = loss_G_GAN + loss_G_loss
                loss_G.backward()
                optimizer_G.step()
                
                with torch.no_grad():
                    # compute average loss
                    avgCostG += loss_G.detach() / batches
                    avgCostD += loss_D.detach() / batches
                    # compute F1-score
                    currF1 = computeF1(fake_gt.squeeze(1), gt, args)
                    avgF1 += currF1 / batches
                
                loss_dict = {"current_loss_G" : loss_G.item(), "current_loss_D" : loss_D.item()}
                pbar.set_postfix(loss_dict)
                pbar.update()
        
        schedulerG.step(loss_G)
        schedulerD.step(loss_D)
        lossesG.append(avgCostG)
        lossesD.append(avgCostD)
        F1s.append(avgF1)
        print()
        print('epoch ' + str(epoch + 1) + ' average D_loss:', avgCostD)
        print('epoch ' + str(epoch + 1) + ' average G_loss:', avgCostG)
        print('epoch ' + str(epoch + 1) + ' average F1-score:', avgF1)

        if (epoch+1) % 5 == 0:
            torch.save(G.state_dict() , os.path.join(args.model_save_dir,"G_%d"%(epoch+start_epoch)))
            torch.save(D.state_dict() , os.path.join(args.model_save_dir,"D_%d"%(epoch+start_epoch)))

    time_elapsed = time() - t0
    print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s.')

    return G

#To generate the submission file over the test dataset
def test_pred(args , G , device = "cpu"): 
    ids = [int(filename.split('_')[1].split('.')[0]) for filename in os.listdir(args.test_path)]         
    # predicting on testset

    testSet = eval(args.image_loader)(args, mode='test')
    testLoader = torch.utils.data.DataLoader(dataset = testSet, batch_size=args.batch_size, shuffle=False)
    
    G.eval()
    images = []
    for x in tqdm(testLoader):
        if(isinstance(x, list) or isinstance(x, tuple)):
            img, another = x
            img = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
            another = another.permute(0,3,1,2).type(torch.FloatTensor).to(device)
            x = [img , another]
        else:
            x = x.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
        
        with torch.no_grad():
            # removing abundant dimension
            pred = G(x).squeeze(1)

        for i in range(len(pred)):
            A = pred[i].cpu().detach().numpy()
            # thresholding
            A[A < 0.5] = 0
            images.append(A)
        resultSubmission(args.sub_CSV_path, images, args, ids)  

def test_visualize(args, G , device = "cpu"):
    
    if args.save and not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    testSet = eval(args.image_loader)(args, mode=args.visualize_dataset)
    testLoader = torch.utils.data.DataLoader(dataset = testSet, batch_size=1, shuffle=(not args.save))
    
    file_names = os.listdir(args.test_path)


    G.eval()
    with tqdm(total=len(testLoader)) as pbar:
        for idx , x in enumerate(testLoader):
        
            if args.visualize_dataset == "train":
                x, gt = x
            
            if(isinstance(x, list) or isinstance(x, tuple)):
                img, another = x
                img1 = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
                another1 = another.permute(0,3,1,2).type(torch.FloatTensor).to(device)
                x = [img1 , another1]
            else:
                img = x
                x = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
            
            with torch.no_grad():
                pred = G(x).squeeze(1)
                pred = pred[0]
                pred = torch.round(pred)
                img = img[0]
                if args.G == "UNetBranched":
                    another = another[0]

                if not args.save:
                    if(args.G == "UNetBranched"):
                        if args.visualize_dataset == "train":
                            fig , axs = plt.subplots(1 ,4)
                            axs[3].imshow(gt[0])
                            axs[3].set_title("ground truth")

                        else:
                            fig , axs = plt.subplots(1 ,3)

                        axs[0].imshow(img)
                        axs[0].set_title("RGB image")
                        axs[1].imshow(another)
                        axs[1].set_title("another image")
                        axs[2].imshow(pred.cpu())
                        axs[2].set_title("prediction")

                        plt.show()

                    else:
                        if args.visualize_dataset == "train":
                            _ , axs = plt.subplots(1 ,3)
                            axs[2].imshow(gt[0])
                            axs[2].set_title("ground truth")
                        else:
                            _ , axs = plt.subplots(1 ,2)

                        axs[0].imshow(img)
                        axs[0].set_title("RGB image")
                        axs[1].imshow(pred.cpu())
                        axs[2].set_title("prediction")

                        plt.show()
            
            if args.save:
                pred = pred.cpu().numpy()
                pred = (pred*255).astype(np.uint8)
                pred = Image.fromarray(pred).convert('L')
                pred.save(os.path.join(args.save_dir , file_names[idx]))

            if idx%args.visualize_samples == args.visualize_samples - 1 and not args.save:
                break
            pbar.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # arguments for models
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch sizes')
    parser.add_argument('--loss', type=str, default='IoULoss', help='the loss function for generator')
    parser.add_argument('--GANloss', type=str, default='BCEWithLogitsLoss', help='the loss function for generator')
    parser.add_argument('--G', type=str, default='XceptionUnetPlusPlus', help='choose a generator')
    parser.add_argument('--D', type=str, default='NLayerDiscriminator', help='choose a discriminator')
    parser.add_argument("--image_loader", type=str , default="ImageLoader", help="Which dataloader class to use")
    parser.add_argument("--model_save_dir", type=str , default="save/", help="Folder to save the models")
    parser.add_argument("--rotate_wrapper" , type=bool , default=False , help="predict with rotate_wrapper")
    parser.add_argument("--secondary_image" , type= str , default="graphCut", help = "To switch between using graphCut or hough transform")

    # paths
    parser.add_argument('--test_path', type=str, default='./data/test_images/')
    parser.add_argument('--sub_CSV_path', type=str, default='./output/submission.csv')
    parser.add_argument('--train_path', type=str, default='./data/trainingCr/images/')
    parser.add_argument('--gt_path', type=str, default='./data/trainingCr/groundtruth/')
    parser.add_argument('--train_mask_path', type=str, default='./data/prediction/groundtruthGC_0.1/', help='GC train-masks for UnetBranched')
    parser.add_argument('--test_mask_path', type=str, default='./data/prediction/XceptionUnet++GAN_cropped_rotate_49epoch_GC_0.1/', help='GC test-masks for UnetBranched')
    parser.add_argument('--output_path', type=str, default='./output/')
    
    # optimizing arguments
    parser.add_argument('--beta1', type=float, default=0.5, help='first decaying parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='second decaying parameter')
    parser.add_argument('--gamma', type=float, default=0.2, help='lr=gamma * lr')
    parser.add_argument('--lr_G', type=float, default=0.0002, help='Learning rate for generator')
    parser.add_argument('--lr_D', type=float, default=0.0002, help='Learning rate for discriminator')
    parser.add_argument('--lambda_loss', type=float, default=100.0, help='weight for loss')

    parser.add_argument("--only_pred" , type=bool , default=False , help="To not train the model. Model path must be provided")
    parser.add_argument("--G_path" , type=str , default = None , help="Path to pretrained G")
    parser.add_argument("--D_path" , type=str , default = None , help="Path to pretrained D")
    parser.add_argument("--start_epoch" , type=int , default = 0 , help="Number of epochs for pretrained G")
    
    parser.add_argument("--visualize" ,type=bool , default=False , help="Visualize predictions on the test dataset")
    parser.add_argument("--visualize_samples" ,type=int , default=5 , help="Number of samples to Visualize")
    parser.add_argument("--visualize_dataset" ,type=str , default="test" , help="Dataset to sample to Visualize")
    parser.add_argument("--save" , type=bool , default=False , help="To save the predictions on the test set")
    parser.add_argument("--save_dir" , type=str , default=None , help="Path to save the predictions on the test set")
    
    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.rotate_wrapper:
        G = RotationWrapperModel(base_model=eval(args.G)()).to(device)
    else:
        G = eval(args.G)().to(device)
    if args.visualize:
        G.load_state_dict(torch.load(args.G_path))
        test_visualize(args , G , device= device)
        exit()
    elif args.only_pred:
        G.load_state_dict(torch.load(args.G_path))
    else:
        D = eval(args.D)().to(device)
        if args.G_path is not None and args.D_path is not None:
            print("Loading pretrained G")
            G.load_state_dict(torch.load(args.G_path))
            D.load_state_dict(torch.load(args.D_path))
        G = run(args=args,G=G , D=D, start_epoch = args.start_epoch , device= device)


        
    test_pred(args,G , device= device)
