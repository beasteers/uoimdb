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


def cache_images(sample, filter, timer_every=50, n=None):
    # load a list of all srcs referenced in the specified random samples    
    sample_srcs = gather_random_sample_srcs(sample)    
    
    # export each image, skipping ones that exist already
    print('Saving images to {}... {} cached images exist there currently.'.format(
        imdb.cfg.IMAGE_CACHE_LOCATION, len(glob.glob(os.path.join(imdb.cfg.IMAGE_CACHE_LOCATION, '*')))))

    if n is not None:
        print('Only caching the first {}/{} images'.format(n, len(sample_srcs)))
        sample_srcs = sample_srcs[n]

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
    for f in glob.glob(os.path.join(random_sample_dir, '{}.csv'.format(sample))):
        if os.path.isfile(f):
            os.remove(f)


def create_sample(sample):
    pass


def gather_random_sample_srcs(sample):
    sample_srcs = []
    for f in glob.glob(os.path.join(random_sample_dir, '{}.csv'.format(sample))):
        idx = pd.read_csv(f, index_col='src').index
        sample_srcs.append(idx)
        print('{} srcs from {}.'.format(len(idx), f))

    sample_srcs = np.unique(np.concatenate(sample_srcs))
    print('In total, {} unique srcs from random samples matching "{}".'.format(len(sample_srcs), sample))
    return sample_srcs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('action', help='create|delete|cache random sample')
    parser.add_argument('sample', help='the random sample to use. can be a glob pattern (e.g. sample-*).', default=None)
    parser.add_argument('-f', '--filter', help='the image filter to use. Can be {}'.format(
        '|'.join(['"'+f+'"' for f in image_processor.filters.keys()])), default='Background Subtraction (mean)')
    parser.add_argument('--timer', help='After how many images should we print out the time statistics.', default=10)
    parser.add_argument('-n', help='Cache the first n images. Useful for testing.', default=None)
    args = parser.parse_args()

    if args.action == 'create':
        raise NotImplementedError('Not implemented yet. sorry. Use the tagging app API for the time being.')

    elif args.action == 'delete':
        delete_sample(args.sample)

    elif args.action == 'cache':
        cache_images(args.sample, filter=args.filter, timer_every=args.timer, n=args.n)

    elif args.action == 'list':
        gather_random_sample_srcs(args.sample)

    else:
        raise ValueError('Action not found.')
    


