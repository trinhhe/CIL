from sklearn.utils import tosequence
from unet import *
from models_DLinkNet import *
from xceptionUnetPlusPlus import *
from RotationWrapperModel import *
from PIL import Image
import matplotlib.pyplot as plt
from imageOperations import *
import argparse
from dataLoader import ImageLoader, ImageLoaderAndAnother, ImageLoader3Channel
import torch.optim as optim
from losses import *
from submissionWriting import *
from torch.utils import data

from tqdm import tqdm

def run(args ,model, start_epoch = 0 , device = "cpu"):
    # build dataset
    trainSet = eval(args.image_loader)(args, mode='train', resize=True , height=512 , width=512)
    
    # build data loader
    trainLoader = torch.utils.data.DataLoader(dataset = trainSet, batch_size = args.batch_size, shuffle=True)
    
    # build model and train

    print(model)
    criterion = eval(args.loss)()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    # learning rate decay
    losses = []

    F1s = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.gamma, patience=5, threshold=0.001, eps=1e-06, verbose=True)
    # start training
    for epoch in tqdm(range(args.epochs)):
        avgCost = 0
        avgF1 = 0
        batches = len(trainLoader)
        print(batches)
        model.train()

        with tqdm(total=batches) as pbar:
            for _ , (img, gt) in enumerate(trainLoader):
                img = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
                
                gt = torch.round(gt)
                gt = gt.type(torch.FloatTensor).to(device)
               
                out = model(img).squeeze(1)
                
                # compute loss
                loss = criterion(out, gt)
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    # compute average loss
                    avgCost += loss.detach() / batches
                    # compute F1-score
                    currF1 = computeF1(out, gt, args)
                    avgF1 += currF1 / batches
                
                loss_dict = {"current_loss" : loss.item()}
                pbar.set_postfix(loss_dict)
                pbar.update()
        
        scheduler.step(loss)
        losses.append(avgCost)
        F1s.append(avgF1)
        print()
        print('epoch ' + str(epoch + 1) + ' average loss:', avgCost)
        print('epoch ' + str(epoch + 1) + ' average F1-score:', avgF1)

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict() , os.path.join(args.model_save_dir,"model_%d"%(epoch+start_epoch)))

    return model

#To get the loss of the model over the training set. To confirm that the correct model is used            
def train_loss(args , model , device = "cpu"):
    trainSet = eval(args.image_loader)(args, mode='train', resize=True , height=512 , width=512)
    # build data loader
    trainLoader = torch.utils.data.DataLoader(dataset = trainSet, batch_size = args.batch_size, shuffle=True)
    
    # build model
    criterion = eval(args.loss)()

    # start  getting training loss
    avgCost = 0
    avgF1 = 0
    batches = len(trainLoader)
    model.eval()
    for (img, gt) in tqdm(trainLoader):
        img = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
        gt = gt.type(torch.FloatTensor).to(device)
        out = model(img).squeeze(1)
        # compute loss
        loss = criterion(out, gt)
        with torch.no_grad():
            # compute average loss
            avgCost += loss.detach() / batches
             # compute F1-score
            currF1 = computeF1(out, gt, args)
            avgF1 += currF1 / batches
        
    print()
    print(' average loss:', avgCost)
    print(' average F1-score:', avgF1)

#To generate the submission file over the test dataset
def test_pred(args , model , device = "cpu"): 
    ids = [int(filename.split('_')[1].split('.')[0]) for filename in os.listdir(args.test_path)]         
    # predicting on testset

    testSet = eval(args.image_loader)(args, mode='test')
    testLoader = torch.utils.data.DataLoader(dataset = testSet, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    images = []
    for img in tqdm(testLoader):
        img = img.permute(0, 3, 1, 2).to(device)
        with torch.no_grad():

            # removing abundant dimension
            pred = model(img).squeeze(1)

        for i in range(len(pred)):
            A = pred[i].cpu().detach().numpy()
            # thresholding
            A[A < 0.5] = 0
            images.append(A)
        resultSubmission(args.sub_CSV_path, images, args, ids)  

def test_visualize(args, model , device = "cpu"):
    
    if args.save and not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    testSet = eval(args.image_loader)(args, mode=args.visualize_dataset , resize=True ,width=512 , height=512)
    testLoader = torch.utils.data.DataLoader(dataset = testSet, batch_size=1, shuffle=(not args.save))
    
    file_names = os.listdir(args.test_path)


    model.eval()
    with tqdm(total=len(testLoader)) as pbar:
        for idx , img in enumerate(testLoader):

            if args.visualize_dataset == "train":
                img , gt = img

            img_1 = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
            
            with torch.no_grad():
                pred = model(img_1).squeeze(1)
                pred = pred[0]
                pred = torch.round(pred)
                img = img[0]
            
                if not args.save:
                    if args.visualize_dataset == "train":
                        _ , axs = plt.subplots(1 ,3)
                        axs[2].imshow(gt[0])
                    else:
                        _ , axs = plt.subplots(1 ,2)

                    axs[0].imshow(img)
                    axs[1].imshow(pred.cpu())

                    plt.show()
            
            if args.save:
                pred = pred.cpu().numpy()
                pred = (pred*255).astype(np.uint8)
                pred = Image.fromarray(pred).convert('RGB')
                pred.save(os.path.join(args.save_dir , file_names[idx]))

            if idx%args.visualize_samples == args.visualize_samples - 1 and not args.save:
                break
            pbar.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # arguments for model
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch sizes')
    parser.add_argument('--loss', type=str, default='IoULoss', help='the loss function')
    parser.add_argument('--model', type=str, default='UNet', help='choose a model')
    parser.add_argument("--image_loader", type=str , default="ImageLoader", help="Which dataloader class to use")
    parser.add_argument("--model_save_dir", type=str , default="save/", help="Folder to save the models")
    parser.add_argument("--rotate_wrapper" , type=bool , default=False , help="predict with rotate_wrapper")
    # paths
    parser.add_argument('--test_path', type=str, default='./data/test_images/')
    parser.add_argument('--sub_CSV_path', type=str, default='./output/submission.csv')
    parser.add_argument('--train_path', type=str, default='./data/training/images/')
    parser.add_argument('--gt_path', type=str, default='./data/training/groundtruth/')
    parser.add_argument('--output_path', type=str, default='./output/')
    
    # optimizing arguments
    parser.add_argument('--beta1', type=float, default=0.9, help='first decaying parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='second decaying parameter')
    parser.add_argument('--gamma', type=float, default=0.2, help='lr=gamma * lr')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    parser.add_argument("--only_pred" , type=bool , default=False , help="To not train the model. Model path must be provided")
    parser.add_argument("--model_path" , type=str , default = None , help="Path to pretrained model")
    parser.add_argument("--start_epoch" , type=int , default = 0 , help="Number of epochs for pretrained model")
    
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
        model = RotationWrapperModel(base_model=eval(args.model)()).to(device)
    else:
        model = eval(args.model)().to(device)
    if args.visualize:
        model.load_state_dict(torch.load(args.model_path))
        test_visualize(args , model , device= device)
        exit()
    elif args.only_pred:
        model.load_state_dict(torch.load(args.model_path))
        # train_loss(args, model , device= device)
    else:
        if args.model_path is not None:
            print("Loading pretrained model")
            model.load_state_dict(torch.load(args.model_path))
        model = run(args=args,model=model , start_epoch = args.start_epoch , device= device)

  
    test_pred(args,model , device= device)
