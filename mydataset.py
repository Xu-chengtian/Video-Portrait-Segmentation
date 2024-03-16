from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms

# rewrite Dataset Class for tiktok dataset
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        # init fuction
        # read image, prior mask and mask path from txt file.(generate in dataset_prepare.py)
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            paths = line.split()
            imgs.append((paths[0], paths[1], paths[2]))
            self.imgs = imgs 
            # set image pre-process method
            self.transform = transform
            self.target_transform = target_transform
    def __getitem__(self, index):
        # return the image, prior mask and mask by index
        transform = transforms.Compose([
            transforms.Resize(572),
            transforms.ToTensor(),
        ])
        image, prior_mask, mask = self.imgs[index]
        image = transform(Image.open(image))
        prior_mask = transform(Image.open(prior_mask))
        combine = torch.cat([image, prior_mask],dim=0)
        mask = transform(Image.open(mask))
        # pre-process
        if self.transform is not None:
            combine = self.transform(combine)
            mask = self.transform(mask)
        return combine, mask
    def __len__(self):
        # return the size of dataset
        return len(self.imgs)