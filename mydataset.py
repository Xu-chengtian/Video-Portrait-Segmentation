from PIL import Image
from torch.utils.data import Dataset

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
        image, prior_mask, mask = self.imgs[index]
        image = Image.open(image)
        prior_mask = Image.open(prior_mask).convert('L')
        mask = Image.open(mask).convert('L')
        # pre-process
        if self.transform is not None:
            image = self.transform(image)
        return image, prior_mask, mask
    def __len__(self):
        # return the size of dataset
        return len(self.imgs)