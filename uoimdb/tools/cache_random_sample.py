import os
import cv2
import glob
import numpy as np
import pandas as pd

import uoimdb as uo
from uoimdb.tagging.image_processing import ImageProcessor

# load image database
imdb = uo.uoimdb()
image_processor = ImageProcessor(imdb)
random_sample_dir = os.path.join(imdb.cfg.DATA_LOCATION, 'random_samples')


def cache_images(sample, filter, timer_every):
    # load a list of all srcs referenced in the specified random samples    
    sample_srcs = []
    for f in glob.glob(os.path.join(random_sample_dir, '{}.csv'.format(sample))):
        idx = pd.read_csv(f, index_col='src').index
        sample_srcs.append(idx)
        print('{} srcs from {}.'.format(len(idx), f))

    sample_srcs = np.unique(np.concatenate(sample_srcs))
    print('Gathered {} unique srcs from random samples matching "{}".'.format(len(sample_srcs), sample))
    
    
    # export each image, skipping ones that exist already
    print('Saving images to {}... {} cached images exist there currently.'.format(
        imdb.cfg.IMAGE_CACHE_LOCATION, len(glob.glob(os.path.join(imdb.cfg.IMAGE_CACHE_LOCATION, '*')))))

    for src in uo.utils.timer(sample_srcs, every=timer_every, what='loop'):

        cache_filename = image_processor.cache_filename(src, args.filter)
        if not os.path.isfile(cache_filename):

            img = image_processor.process_image(src, args.filter)

            if img is None:
                print('Failed creating {}.'.format(cache_filename))
            else: 
                cv2.imwrite(cache_filename, img)

    print('Done. {} now contains {} images.'.format(
        imdb.cfg.IMAGE_CACHE_LOCATION, len(glob.glob(os.path.join(imdb.cfg.IMAGE_CACHE_LOCATION, '*')))))



def delete_sample(sample):
    for f in glob.glob(os.path.join(random_sample_dir, '{}.csv'.format(name))):
        if os.path.isfile(f):
            os.remove(f)


def create_sample(sample):
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('action', help='create|delete|cache random sample')
    parser.add_argument('sample', help='the random sample to use. can be a glob pattern (e.g. sample-*).')
    parser.add_argument('-f', '--filter', help='the image filter to use. Can be {}'.format(
        '|'.join(['"'+f+'"' for f in image_processor.filters.keys()])), default='Background Subtraction (mean)')
    parser.add_argument('--timer', help='After how many images should we print out the time statistics.', default=10)
    args = parser.parse_args()

    # if args.action == 'create':
    #     pass

    # if args.action == 'delete':
    #     delete_sample(args.sample)

    # if args.action == 'cache':
        
    cache_images(args.sample, filter=args.filter, timer_every=args.timer)
    

