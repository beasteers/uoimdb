import os
import glob
import uoimdb as uo



def build_pipeline(imdb):
    pipeline = (imdb.pipeline()
                   .grey()
                   .bgsub(window=window, mode='valid')
                   .scale()
                   .invert()
                   .fake_crop()
                   .cmap()
                   .astype('uint8'))
    return pipeline



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('sample', help='the random sample to use. can be a glob pattern (e.g. sample-*).')
    args = parser.parse_args()
    
    # load a list of all srcs referenced in the specified random samples
    random_sample_path = os.path.join(imdb.cfg.DATA_LOCATION, 'random_samples', '{}.csv'.format(args.sample))
    sample_srcs = np.unique(np.concatenate([
        pd.read_csv(f, index_col='src').index
        for f in glob.glob(random_sample_path)
    ]))
    
    # load image database
    imdb = uo.uoimdb()
    
    # build and run image pipeline
    pipeline = build_pipeline(imdb)
    for save_path in pipeline.timer(50).save_images(imdb.cfg.IMAGE_CACHE_LOCATION):
        pass
