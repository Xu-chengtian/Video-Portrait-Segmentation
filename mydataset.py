from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms

# rewrite Dataset Class for tiktok dataset
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None, scale = 0.5):
        # init fuction
        # read image, prior mask and mask path from txt file.(generate in dataset_prepare.py)
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            paths = line.split()
            imgs.append((paths[0], paths[1], paths[2])) 
        # set image pre-process method
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.scale = scale
        assert len(imgs) > 0, "There is no valid dataset in the provide directory"
        img = imgs[0][0]
        img = Image.open(img)
        w,h = img.size
        self.newW, self.newH = int(self.scale * w), int(self.scale * h)
        assert self.newW > 0 and self.newH > 0, 'Scale is too small, resized images would have no pixel'
    def __getitem__(self, index):
        # return the image, prior mask and mask by index
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image, prior_mask, mask = self.imgs[index]
        image = transform(Image.open(image).resize((self.newW, self.newH), resample=Image.BICUBIC))
        prior_mask = transform(Image.open(prior_mask).resize((self.newW, self.newH), resample=Image.NEAREST))
        image = torch.cat([image, prior_mask],dim=0)
        mask = transform(Image.open(mask).resize((self.newW, self.newH), resample=Image.NEAREST))
        # pre-process
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    def __len__(self):
        # return the size of dataset
        return len(self.imgs)