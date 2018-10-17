import os
import sys
import cv2
import glob
import hashlib
import numpy as np
import pandas as pd
from filelock import FileLock
from multiprocessing.pool import ThreadPool

import uoimdb as uo
from uoimdb.tagging.image_processing import ImageProcessor
from uoimdb.tagging.app import user_col

import traceback




class RandomSamples(object):
    def __init__(self, imdb, image_processor=None):
        self.imdb = imdb
        self.image_processor = image_processor
        self.random_sample_dir = imdb.cfg.RANDOM_SAMPLE_LOCATION
        uo.utils.ensure_dir(self.random_sample_dir)

        self.samples = {}
        self._permutations = {} # stores the shuffle order for each user+sample


    def load_samples(self, name='*'):
        for f in glob.glob(os.path.join(self.random_sample_dir, '{}.csv'.format(name))):
            self.load_sample(f) # loads into self.samples[name]


    def load_sample(self, filename):
        df = pd.read_csv(filename, index_col='src').fillna('')

        name = os.path.splitext(os.path.basename(filename))[0]

        for user in self.imdb.cfg.USERS:
            # create columns that don't exist. resilient to new users
            col = user_col('status', user)
            if not col in df.columns:
                df[col] = ''

            # create sample shuffle order for all users     
            perm_id = user_col(name, user)
            seed = int(hashlib.md5(perm_id.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
            self._permutations[perm_id] = np.random.RandomState(seed=seed).permutation(len(df))

        self.samples[name] = df



    def user_sample_order(self, name, user=None):
        order = self._permutations[user_col(name, user=user)] # 
        return self.samples[name].iloc[order]



    def create_sample(self, sample, time_range=None, distance_to_gap=None, overlap_existing=None, 
                        n_samples=None, overlap_ratio=None, ):
        imdb = self.imdb
        df = imdb.df

        # filter by time range
        if time_range is None:
            time_range = imdb.cfg.FILTER_TIME_RANGE
        if time_range:
            morning, evening = time_range
            if morning:
                df = df[df.date.dt.hour >= morning]
            if evening:
                df = df[df.date.dt.hour < evening]

        # dont take images close to a time gap
        if distance_to_gap is None:
            distance_to_gap = int(imdb.cfg.DISTANCE_FROM_GAP)
        if distance_to_gap:
            df = df[df.distance_to_gap >= distance_to_gap]

        # create sample independent of other samples
        if overlap_existing is not None:
            for _, sample in self.samples.items():
                df = df.drop(index=sample.index, errors='ignore')

        if not len(df):
            return False

        names = []
        if n_samples: # create several overlapping samples
            overlap_ratio = float(overlap_ratio or imdb.cfg.SAMPLE_OVERLAP_RATIO)
            n_needed = int(n * (n_samples - (n_samples - 1) * overlap_ratio))

            full_sample = df.sample(n=min(n_needed, len(df))).index

            for i in range(n_samples):
                start = int(i * n * (1 - overlap_ratio))
                sample = full_sample[start:start + n]
                if len(sample):
                    sample_name = '{}-{}'.format(name, i+1)
                    self._create_sample_file(sample_name, sample)
                    names.append(sample_name)

        else: # create a single sample
            sample = df.sample(n=min(n, len(df))).index
            self._create_sample_file(name, sample)
            names.append(name)

        return names


    def _create_sample_file(self, name, sample):
        filename = os.path.join(self.random_sample_dir, '{}.csv'.format(name))

        # I know this isn't the most efficient way, but I'm creating user cols in load_..
        pd.DataFrame(columns=[], index=sample).to_csv(filename) 
        self.load_sample(filename)



    def save_user_sample(self, name):
        filename = os.path.join(self.random_sample_dir, '{}.csv'.format(name))
        my_sample = self.samples[name]

        with FileLock(filename + '.lock', timeout=3): # handle concurrent access
            # load existing sample
            sample = pd.read_csv(filename, index_col='src')

            # update all user columns
            for col in my_sample.columns: 
                if col.startswith(user_col('')):
                    sample[col] = my_sample[col]

            sample.to_csv(filename)


    def delete_sample(self, sample):
        for f in glob.glob(os.path.join(self.random_sample_dir, '{}.csv'.format(sample))):

            name = os.path.splitext(os.path.basename(f))[0]
            if name in self.samples:
                del self.samples[name]

            if os.path.isfile(f):
                os.remove(f)



    def gather_sample_srcs(self, sample, window=None):
        sample_srcs = []
        for f in glob.glob(os.path.join(self.random_sample_dir, '{}.csv'.format(sample))):
            idx = pd.read_csv(f, index_col='src').index
            sample_srcs.append(idx)

            print('{} srcs from {}.'.format(len(idx), f))

        sample_srcs = np.unique(np.concatenate(sample_srcs))
        print('In total, {} unique srcs from random samples matching "{}".'.format(len(sample_srcs), sample))

        if window:
            sample_srcs = np.unique(np.concatenate([
                self.imdb.around_src(src, window)
                for src in sample_srcs
            ]))
            print('Including a window of {}, {} unique srcs.'.format(window, len(sample_srcs), sample))
        return sample_srcs


    def cache_images(self, sample_srcs, filter=None, timer_every=50, n=None, offset=None, recompute=False, pool=False, raise_errors=False):
        # load a list of all srcs referenced in the specified random samples    
        imdb, image_processor = self.imdb, self.image_processor
        if image_processor is None:
            self.image_processor = ImageProcessor(imdb)

        filter = filter or imdb.cfg.DEFAULT_FILTER

        # order images by date for more efficient loading
        sample_srcs = imdb.df.date[sample_srcs].sort_values().index
        
        # create directory if it's missing
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

    



if __name__ == '__main__':
    # load image database
    imdb = uo.uoimdb()
    image_processor = ImageProcessor(imdb)
    sampler = RandomSamples(imdb, image_processor)
    
    
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
        random_samples.delete_sample(args.sample)

    elif args.action == 'cache':
        if not os.isatty(0):
            print('Gathering srcs from stdin...')
            if args.recompute:
                print("Because the recompute flag was used, it's assumed that the images are specified using their paths in the image cache directory.")
            
            sample_srcs = np.array([
                image_processor.inverse_cache_filename(l) if args.recompute else l
                for l in uo.utils.yield_lines(sys.stdin) ])
            
            print('Gathered {} images.'.format(len(sample_srcs)))
            print(sample_srcs[:5])
            
        else:
            print('Starting caching of {} using the "{}" filter...'.format(args.sample, args.filter))
            sample_srcs = random_samples.gather_sample_srcs(imdb, args.sample, window) 
        
        random_samples.cache_images(image_processor, sample_srcs, filter=args.filter, n=args.n, offset=args.offset, 
                                    recompute=args.recompute, pool=args.pool, timer_every=args.timer, raise_errors=args.errors)

    elif args.action == 'list':
        random_samples.gather_sample_srcs(imdb, args.sample, window=window)

    else:
        raise ValueError('Action {} not found.'.format(args.action))
  


