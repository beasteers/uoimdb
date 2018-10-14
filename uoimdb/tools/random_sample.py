import os
import sys
import cv2
import glob
import numpy as np
import pandas as pd
from multiprocessing.pool import ThreadPool

import uoimdb as uo
from uoimdb.tagging.image_processing import ImageProcessor

import traceback



def cache_images(image_processor, sample_srcs, filter=None, timer_every=50, n=None, offset=None, recompute=False, pool=False, raise_errors=False):
    # load a list of all srcs referenced in the specified random samples    
    imdb = image_processor.imdb
    filter = filter or imdb.cfg.DEFAULT_FILTER

    # order images by date for more efficient loading
    sample_srcs = imdb.df.date[sample_srcs].sort_values().index
    
    uo.utils.ensure_dir(imdb.cfg.IMAGE_CACHE_LOCATION)

    # export each image, skipping ones that exist already
    print('Saving images to {}... {} cached images exist there currently.'.format(
        imdb.cfg.IMAGE_CACHE_LOCATION, len(glob.glob(os.path.join(imdb.cfg.IMAGE_CACHE_LOCATION, '*')))))

    if offset is not None:
        print('Starting at image {}.'.format(offset))
        sample_srcs = sample_srcs[offset:]
    if n is not None:
        print('Only caching {} images.'.format(n))
        sample_srcs = sample_srcs[:n]

    if pool:
        if pool is True:
            pool = 30
        print('loading images using {} threads'.format(pool))
        pool = ThreadPool(pool)
        
    for i, src in enumerate(uo.utils.timer(sample_srcs, every=timer_every, what='loop')):
        cache_filename = image_processor.cache_filename(src, filter, ext=imdb.cfg.IMAGE_CACHE_EXT)
        if recompute or not os.path.isfile(cache_filename):
            try:
                img = image_processor.process_image(src, filter, pool=pool)
            except Exception as e:
                print('!!! Error creating image {}: {}'.format(i, src))
                if raise_errors:
                    raise e
                else:
                    traceback.print_exc()
                    continue
                

            if img is None:
                print('Failed creating {}.'.format(cache_filename))
            else: 
                cv2.imwrite(cache_filename, img)

    print('Done. {} now contains {} images.'.format(
        imdb.cfg.IMAGE_CACHE_LOCATION, len(glob.glob(os.path.join(imdb.cfg.IMAGE_CACHE_LOCATION, '*')))))

    if pool:
        pool.close()
        pool.join()
    

def delete_sample(imdb, sample):
    random_sample_dir = os.path.join(imdb.cfg.DATA_LOCATION, 'random_samples')
    for f in glob.glob(os.path.join(random_sample_dir, '{}.csv'.format(sample))):
        if os.path.isfile(f):
            os.remove(f)


def create_sample(imdb, sample):
    pass


def gather_random_sample_srcs(imdb, sample, window=None):
    random_sample_dir = os.path.join(imdb.cfg.DATA_LOCATION, 'random_samples')
    sample_srcs = []
    for f in glob.glob(os.path.join(random_sample_dir, '{}.csv'.format(sample))):
        idx = pd.read_csv(f, index_col='src').index
        sample_srcs.append(idx)
        print('{} srcs from {}.'.format(len(idx), f))

    sample_srcs = np.unique(np.concatenate(sample_srcs))
    print('In total, {} unique srcs from random samples matching "{}".'.format(len(sample_srcs), sample))

    if window:
        sample_srcs = np.unique(np.concatenate([
            imdb.around_src(src, window)
            for src in sample_srcs
        ]))
        print('Including a window of {}, {} unique srcs.'.format(window, len(sample_srcs), sample))
    return sample_srcs


def src_from_cache_fn(fn):
    '''Reverse of ImageProcessor.cache_filename(). 
        NOTE... This is hard-coded to our specific format. Need to generalize this later. '''
    fn = os.path.basename(fn)
    filter, idx = fn.split(',', 1)
    src = os.path.splitext(idx.replace(',', '/'))[0] + '.png'
    return src


if __name__ == '__main__':
    # load image database
    imdb = uo.uoimdb()
    image_processor = ImageProcessor(imdb)
    
    
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('action', help='create|delete|cache random sample')
    parser.add_argument('sample', help='the random sample to use. can be a glob pattern (e.g. sample-*).', default=None)
    parser.add_argument('-f', '--filter', help='the image filter to use. Can be {}'.format(
        '|'.join(['"'+f+'"' for f in image_processor.filters.keys()])), default=imdb.cfg.DEFAULT_FILTER)
    parser.add_argument('--timer', help='After how many images should we print out the time statistics.', default=10, type=int)
    parser.add_argument('-w', '--window', help='Also cache images surrounding the sample images.', nargs='?', default=None, const=imdb.cfg.SAMPLE_WINDOW)
    parser.add_argument('-n', help='Cache the first n images. Useful for testing.', default=None, type=int)
    parser.add_argument('--offset', help='Which index to start at.', default=None, type=int)
    parser.add_argument('--recompute', help="Don't skip if the image already exists", action='store_true')
    parser.add_argument('-e', '--errors', help="If an error occurs, raise it. Otherwise, continue on to the next image.", action='store_true')
    parser.add_argument('--pool', help="Use threads to load images", nargs='?', default=None, const=True, type=int)
    args = parser.parse_args()
    

    try:
        window = [int(w) for w in args.window.split(',')]
        if len(window) == 1:
            window = (window[0], window[0])
    except Exception:
        window = args.window

    if args.action == 'create':
        raise NotImplementedError('Not implemented yet. sorry. Use the tagging app API (/random/<name>/create/) for the time being.')

    elif args.action == 'delete':
        delete_sample(imdb, args.sample)

    elif args.action == 'cache':
        if not os.isatty(0):
            print('Gathering srcs from stdin...')
            if args.recompute:
                print("Because the recompute flag was used, it's assumed that the images are specified using their paths in the image cache directory.")
            sample_srcs = np.array([
                src_from_cache_fn(l) if args.recompute else l
                for l in uo.utils.yield_lines(sys.stdin)
            ])
            print('Gathered {} images.'.format(len(sample_srcs)))
            print(sample_srcs[:5])
            
        else:
            print('Starting caching of {} using the "{}" filter...'.format(args.sample, args.filter))
            sample_srcs = gather_random_sample_srcs(imdb, args.sample, window) 
        
        cache_images(image_processor, sample_srcs, filter=args.filter, n=args.n, offset=args.offset, 
                     recompute=args.recompute, pool=args.pool, timer_every=args.timer, raise_errors=args.errors)

    elif args.action == 'list':
        gather_random_sample_srcs(imdb, args.sample, window=window)

    else:
        raise ValueError('Action not found.')
  


