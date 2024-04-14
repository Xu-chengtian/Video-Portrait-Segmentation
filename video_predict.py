import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from logger import setlogger

from Unet import UNet
import cv2

def predict_img(net,
                image,
                prior_mask,
                device,
                out_threshold=0.5):
    net.eval()

    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    image = transform(image)
    
    logger.info(image.shape)
    
    w, h = image.shape[1], image.shape[2]
    newH, newW = int(0.5 * w), int(0.5 * h)

    if type(prior_mask) != torch.Tensor:
        prior_mask = transform(prior_mask)
    
    torch_resize = transforms.Resize([newH, newW])
    image = torch_resize(image)
    prior_mask = torch_resize(prior_mask)
    combine = torch.cat([image, prior_mask],dim=0).unsqueeze(0)
    
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
                transforms.Resize([w, h]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', metavar='FILE',
                        help="Specify the file in which the model is stored")
    
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    
    parser.add_argument('--video_file', '-f', metavar='INPUT',
                        help="Input video file for segmentation")

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    setlogger(logger)
    args = get_args()

    net = UNet(n_channels=4, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    cap = cv2.VideoCapture(args.video_file) if args.video_file else cv2.VideoCapture(0)
    mask = torch.zeros(1, 1, 1)
    while (1):
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        

        # '''
        mask = predict_img(net=net,
                        image=img,
                        prior_mask=mask,
                        out_threshold=args.mask_threshold,
                        device=device)
        # '''
        result = mask_to_image(mask)
        # print(result)
        if args.viz:
            cv2.imshow("capture", np.array(result))
            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break