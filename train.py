#import the Unet Model
from Unet import UNet
from logger import setlogger
from mydataset import MyDataset
from eval import eval_net

#import pytorch
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time

# import wandb
import warnings
import os
import sys
import argparse
import logging

#import evluation module
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss

# fuction trained the model
def train_model(logger, net, device, epochs=5, batch_size=100, lr=0.01, val_percent=0.1, save_cp=True, img_scale=0.5):
    # val_precent: precent of database use for validation
    pre_time=time.strftime('%m%d%H%M%S')
    project_name = 'prior mask' + pre_time
    # # wandb初始化
    # wandb.init(project='Video-Portrait-Segmentation', name=project_name)
    # wandb.config = {"time": pre_time, "batch_size": batch_size, "epochs": epochs,
    #                 "learning rate": lr}
    # wandb.config.update()
    # # 建立日志、模型存储文件夹
    dir_checkpoint = os.path.join(os.getcwd(),'models',project_name)
    os.makedirs(dir_checkpoint)
    os.makedirs(os.path.join(os.getcwd(),'log',project_name))

    dataset = MyDataset(os.path.join(os.getcwd(),'dataset','train.txt'))
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_data_loader = DataLoader(train,shuffle=True,batch_size=batch_size,num_workers=8, pin_memory=True,drop_last=True)
    val_data_loader = DataLoader(val,batch_size=batch_size,num_workers=8, pin_memory=True,drop_last=True)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(epochs):
        net.train()
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch_idx,data in enumerate(train_data_loader):
                combine, mask = data
                                
                combine = combine.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                mask = mask.to(device=device, dtype=mask_type)
                
                optimizer.zero_grad()
                masks_pred = net(combine)
                loss = criterion(masks_pred, mask)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                loss.backward()
                optimizer.step()

                pbar.update(combine.shape[0])
                global_step += 1
                dataset_len = len(dataset)
                a1 = dataset_len // 10
                a2 = dataset_len / 10
                b1 = global_step % a1
                b2 = global_step % a2

                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(net, val_data_loader, device, n_val)
                    if net.n_classes > 1:
                        logger.info('Validation cross entropy: {}'.format(val_score))

                    else:
                        logger.info('Validation Dice Coeff: {}'.format(val_score))

            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logger.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                        dir_checkpoint + f'CP_epoch{epoch + 1}_loss_{str(loss.item())}.pth')
                logger.info(f'Checkpoint {epoch + 1} saved ! loss (batch) = ' + str(loss.item()))
    # wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description='Train prior mask method model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logger = logging.getLogger(__name__)
    logger = setlogger(logger)
    logger.info('Start')
    
    args = get_args()
    
    # using cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    # in prior mask method, the input channel is 4 and we only need one ourput. (portrait)
    net = UNet(n_channels=4, n_classes=1)
    logger.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')
    
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logger.info(f'Model loaded from {args.load}')

    net.to(device)
    
    try:
        train_model(logger=logger,
                    net=net,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'models/INTERRUPTED.pth')
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    logger.info('Finish')