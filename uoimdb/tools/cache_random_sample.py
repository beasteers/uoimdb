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





def build_pipeline(imdb, method, cmap=None, scale=None):
    pipeline = (imdb.pipeline()
                     .grey()
                     .single_bgsub(method=method)
                     .scale(scale)
                     .clip()
                     .invert()
                     .cmap(cmap)
                     .astype('uint8'))
    return pipeline



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('sample', help='the random sample to use. can be a glob pattern (e.g. sample-*).')
    parser.add_argument('-f', '--filter', help='the image filter to use. Can be {}'.format(
      '|'.join(['"'+f+'"' for f in image_processor.filters.keys()])), default='Background Subtraction (mean)')
    parser.add_argument('--timer', help='After how many images should we print out the time statistics.', default=10)
    args = parser.parse_args()

    
    
    # load a list of all srcs referenced in the specified random samples
    random_sample_path = os.path.join(imdb.cfg.DATA_LOCATION, 'random_samples', '{}.csv'.format(args.sample))
    
    sample_srcs = []
    for f in glob.glob(random_sample_path):
      idx = pd.read_csv(f, index_col='src').index
      sample_srcs.append(idx)
      print('{} srcs from {}.'.format(len(idx), f))

    sample_srcs = np.unique(np.concatenate(sample_srcs))
    print('Gathered {} unique srcs from random samples matching "{}"'.format(len(sample_srcs), args.sample))
    
    
    # build and run image pipeline
    print('Saving images to {}... {} cached images exist there currently.'.format(
      imdb.cfg.IMAGE_CACHE_LOCATION, len(glob.glob(os.path.join(imdb.cfg.IMAGE_CACHE_LOCATION, '*')))))

    for src in uo.utils.timer(sample_srcs, every=args.timer, what='loop'):
      cache_filename = image_processor.cache_filename(src, args.filter)

      if not os.path.isfile(cache_filename):
        img = image_processor.process_image(src, args.filter)
        if img is None:
          print('Failed creating {}.'.format(cache_filename))
        else: 
          cv2.imwrite(cache_filename, img)

    print('Done. {} now contains {} images.'.format(
      imdb.cfg.IMAGE_CACHE_LOCATION, len(glob.glob(os.path.join(imdb.cfg.IMAGE_CACHE_LOCATION, '*')))))

