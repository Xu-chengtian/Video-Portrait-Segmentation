import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff, multiclass_dice_coeff


def eval_net(net, loader, device, n_val, amp):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    dice_score = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch_idx,data in enumerate(loader):
            imgs, true_masks = data

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)

            if net.n_classes == 1:
                assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, true_masks, reduce_batch_first=False)
            else:
                assert true_masks.min() >= 0 and true_masks.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                true_masks = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], true_masks[:, 1:], reduce_batch_first=False)
            pbar.update(imgs.shape[0])

    net.train()
    return dice_score / max(n_val / 5, 1)