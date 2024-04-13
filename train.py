from Unet import UNet
from logger import setlogger
from mydataset import MyDataset
from eval import eval_net
from dice_loss import dice_loss

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import wandb
import warnings
import os
import sys
import argparse
import logging

# fuction trained the model
def train_model(logger, project_name, net, device, combine = False, epochs=5, batch_size=100, lr=0.01, frequency=10, val_percent=0.1, img_scale=0.5, amp = False):
    # val_precent: precent of database use for validation
    
    # init wandb
    wb = wandb.init(project='Video-Portrait-Segmentation', name=project_name, resume='allow', anonymous='must')
    wb.config.update(dict(time=time.strftime('%m%d%H%M%S'), combine=combine, epochs=epochs, batch_size=batch_size, learning_rate=lr,
             validation_frequency=frequency, val_percent=val_percent, img_scale=img_scale, amp=amp))

    dir_checkpoint = os.path.join(os.getcwd(), 'models', project_name)
    os.makedirs(dir_checkpoint)
    if combine:
        dataset = MyDataset(os.path.join(os.getcwd(), 'dataset', 'train.txt'), scale = img_scale)
    else:
        dataset = MyDataset(os.path.join(os.getcwd(), 'dataset', 'train_without_combine.txt'), scale = img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator())
    train_data_loader = DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=1, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(val, batch_size=batch_size, num_workers=1, pin_memory=True, drop_last=True)

    logger.info(f'''Starting training:
        project_name:    {project_name}
        Combine:         {combine}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Evaluate freq:   {frequency}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled = amp)
    
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    division_step = (n_train // (frequency * batch_size))
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch_idx,data in enumerate(train_data_loader):
                combine, mask = data       
                combine = combine.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                mask = mask.to(device=device, dtype=mask_type)
                
                masks_pred = net(combine)
                loss = criterion(masks_pred.squeeze(1), mask.squeeze(1).float())
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), mask.squeeze(1).float(), multiclass=False)
                
                optimizer.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(combine.shape[0])
                epoch_loss += loss.item()
                wb.log({
                    'train loss': loss.item(),
                    'step': batch_idx+1,
                    'epoch': epoch
                })
                        
                if division_step == 0 or (batch_idx + 1) % division_step == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        if not (torch.isinf(value) | torch.isnan(value)).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = eval_net(net, val_data_loader, device, n_val, amp)
                    scheduler.step(val_score)

                    logger.info('Validation Dice score: {}'.format(val_score))
                    try:
                        wb.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(combine[0].cpu()),
                            'masks': {
                                'true': wandb.Image(mask[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': batch_idx + 1,
                            'epoch': epoch,
                            **histograms
                        })
                    except:
                        pass

            torch.save(net.state_dict(),
                    dir_checkpoint+'/'+ f'CP_epoch{epoch + 1}_loss_{str(loss.item())}.pth')
            logger.info(f'Checkpoint {epoch + 1} saved ! loss (batch) = ' + str(loss.item()))
    wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description='Train prior mask method model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--combine', action='store_true', default=False,
                        help='using combine of prior mask', dest='combine')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--eval-frequency', type=int, default=10,
                        help='Evaluate Frequency', dest='evaluate_frequency')
    parser.add_argument('-d', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('-w', '--wandb', type=str, default=None, help='input wandb api key')
    return parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logger = logging.getLogger(__name__)
    logger = setlogger(logger)
    logger.info('Start')
    
    args = get_args()
    
    if args.wandb:
        os.environ["WANDB_API_KEY"] = args.wandb
    
    # using cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    # in prior mask method, the input channel is 4 and we only need one ourput. (portrait)
    net = UNet(n_channels=4, n_classes=1, bilinear=True)
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
    
    project_name = 'prior mask' + time.strftime('%m%d%H%M%S')
    
    try:
        train_model(logger = logger,
                    project_name = project_name,
                    net = net,
                    combine = args.combine,
                    epochs = args.epochs,
                    batch_size = args.batchsize,
                    lr = args.lr,
                    frequency = args.evaluate_frequency,
                    device = device,
                    img_scale = args.scale,
                    val_percent = args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'models/INTERRUPTED.pth')
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    logger.info('Finish')