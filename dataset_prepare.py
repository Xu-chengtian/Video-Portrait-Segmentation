import argparse
import os
import logging
from logger import setlogger

def genearte_dataset_txt(logger, path):
    cur_path = os.getcwd()
    dataset_path = os.path.join(path, 'train')
    empty_prior = os.path.join(cur_path, "dataset", "empty_prior.png")
    
    train = open(os.path.join(cur_path,'dataset','train.txt'), 'w+')
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
            train.write(os.path.join(os.path.join(dataset_path, video, 'images', frame)))
            train.write(' ')
            if prev == None:
                train.write(empty_prior)
            else:
                train.write(os.path.join(os.path.join(dataset_path, video, 'masks', prev)))
            prev = frame
            train.write(' ')
            train.write(os.path.join(os.path.join(dataset_path, video, 'masks', frame)))
            train.write('\n')          
    train.close()

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    setlogger(logger)
    logger.info('Start')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, help='provide data set path')
    args = parser.parse_args()
    logger.info(args)
    logger.info("Generate dataset txt for : "+args.path)
    genearte_dataset_txt(logger, args.path)