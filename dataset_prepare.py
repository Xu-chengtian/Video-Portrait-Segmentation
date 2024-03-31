import argparse
import os
import logging
from logger import setlogger
import random

def genearte_dataset_txt(logger, path, scale):
    cur_path = os.getcwd()
    dataset_path = os.path.join(path, 'train')
    empty_prior = os.path.join(cur_path, "dataset", "empty_prior.png")
    
    train = open(os.path.join(cur_path,'dataset','train.txt'), 'w+')
    train_without_combine = open(os.path.join(cur_path,'dataset','train_without_combine.txt'), 'w+')
    videos = os.listdir(dataset_path)
    videos.sort()
    for video in videos:
        if video == '.DS_Store': continue
        logger.info("handeling for folder: "+video)
        video_path = os.path.join(dataset_path, video, 'images')
        frames = os.listdir(video_path)
        prev = None
        frames.sort()
        for frame in frames:
            if frame == '.DS_Store': continue
            add_flag = random.random()<=scale
            image = os.path.join(os.path.join(dataset_path, video, 'images', frame))
            prior = empty_prior if prev == None else os.path.join(os.path.join(dataset_path, video, 'masks', prev))
            prev = frame
            mask = os.path.join(os.path.join(dataset_path, video, 'masks', frame))
            train.write(image + ' ' + prior + ' ' + mask + '\n')
            if add_flag:
                train_without_combine.write(image + ' ' + empty_prior + ' ' + mask + '\n')
    train.close()
    train_without_combine.close()

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    setlogger(logger)
    logger.info('Start')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, help='provide data set path')
    parser.add_argument('-s', '--scale', type=float, default=0.2, help='scale of dataset without prior mask')
    args = parser.parse_args()
    logger.info(args)
    assert args.scale>=0 and args.scale<=1, 'scale should be between 0 and 1'
    logger.info("Generate dataset txt for : "+ args.path)
    genearte_dataset_txt(logger, args.path, args.scale)
    logger.info('End')