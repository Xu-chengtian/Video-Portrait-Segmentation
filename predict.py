import argparse
import logging
from logger import setlogger

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from Unet import UNet
import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask, result):
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Input image')
    image = Image.open(img)
    ax[0].imshow(image)
    ax[1].set_title(f'Input prior mask')
    mask = Image.open(mask)
    ax[1].imshow(mask)
    ax[2].set_title(f'Output mask')
    result = Image.open(result)
    ax[2].imshow(result)
    plt.xticks([]), plt.yticks([])
    plt.show()


def predict_img(net,
                image,
                prior_mask,
                device,
                out_threshold=0.5):
    net.eval()

    transform = transforms.Compose([
            transforms.Resize(572),
            transforms.ToTensor(),
        ])
    image = transform(Image.open(image))
    prior_mask = transform(Image.open(prior_mask))
    combine = torch.cat([image, prior_mask],dim=0).unsqueeze(0)
    
    # print(combine.shape)

    img = combine.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(640),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--image', '-i', metavar='INPUT',
                        help='filenames of input images', required=True)
    parser.add_argument('--mask', '-k', metavar='INPUT',
                        help='filenames of input mask', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT',
                        help='Filenames of ouput images', default = 'output.png')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    setlogger(logger)
    args = get_args()
    image = args.image
    mask = args.mask
    out_file = args.output

    net = UNet(n_channels=4, n_classes=1)

    logger.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logger.info("Model loaded !")

    mask = predict_img(net=net,
                        image=image,
                        prior_mask=mask,
                        out_threshold=args.mask_threshold,
                        device=device)
    
    result = mask_to_image(mask)
    result.save(out_file)

    logger.info("Mask saved to {}".format(out_file))

    if args.viz:
        logger.info("Visualizing results for image {}, close to continue ...".format(image))
        plot_img_and_mask(image, args.mask, out_file)