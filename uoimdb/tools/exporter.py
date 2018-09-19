from __future__ import print_function
import os
import cv2
import glob
import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

# import uoimages as uo
from .. import uoimdb
from .. import utils
from ..utils import progress, timer, ensure_dir
from ..dataset import get_ground_truth



'''Annotations'''

def filename2index(filename):
    '''Because the uo images are stored in subdirectories, we need an id that can be used as a filename.'''
    return filename.replace('/', ',')



def generate_annotation(src, labels, size, date=None, bgsub_params=None, crop=None):
    '''Build the annotation xml file structure
    Arguments:
        src (str): should be the relative path to image
        plumes (pd.DataFrame): plumes for that day
        date (castable to str): timestamp of file.
    '''
    if len(size) == 3:
        height, width, depth = size
    else:
        (height, width), depth = size, 3

    return {
        'folder': 'uoimdb',
        'filename': filename2index(src), 
        'timestamp': date,
        'source': {
            'database': 'Urban Observatory Camera: 1 Metrotech Center',
            'path': src,
        },
        'owner': {
            'name': 'Urban Observatory',
        },
        'size': {
            'width': width,
            'height': height,
            'depth': depth,
        },
        'image_generation_params': bgsub_params,
#           'segmented': 0,
        'object': [
            {
                'name': label.label,
                'pose': '',
                'truncated': 0,
                'difficult': 0, # either include this or set config['use_diff'] = True
                'bndbox': {
                    'xmin': max(1, int(max(label.x1, 0)*width)), # int(max(label.x - label.w / 2, 0) * width),
                    'xmax': max(1, int(min(label.x2, 1)*width)), # int(min(label.x + label.w / 2, 1) * width),
                    'ymin': max(1, int(max(label.y1, 0)*height)), # int(max(label.y - label.h / 2, 0) * height),
                    'ymax': max(1, int(min(label.y2, 1)*height)), # int(min(label.y + label.h / 2, 1) * height),
                },
                # misc meta
                'user': label.get('user'),
                'id': label.id,
                'prev_id': label.get('prev_id'),
                'origin_id': label.get('origin_id')
            }
            for _, label in labels.iterrows()
        ],
    }


def dict2xml(data, root='root', key=None):
    '''Convert dict to xml tree.
    Arguments:
        data (dict): the data to convert
        root (str or xml element): the root element (or element name) to data to.
        key (for internal use): tag name. used for recursion.
    
    For multiple entries with the same name, pass as {'tag' ['hi', 'hello']}
    Renders as:
        <tag>hi</tag>
        <tag>hello</tag>
    '''
    if isinstance(root, str):
        root = ET.Element(root)
        
    if key is not None:
        root = ET.SubElement(root, key)
    
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    dict2xml(v, root=root, key=key)
            else:
                dict2xml(val, root=root, key=key)
    elif data is not None:
        root.text = str(data)
    return root





'''Training Sets'''


def split_dataset(data, splits):
    '''Perform train/test split with an arbitrary number of splits.
    Arguments:
        data (numpy.array, pandas.Series, etc.): something that can take binary indexing.
        splits (list): the splits you want to take. List is normalized to sum to 1.
    Returns:
        (len(split) + 1) splits.
    '''
    norm = sum(splits) * 1. if len(splits) > 1 else 1
    rand = np.random.rand(len(data))
    splits = [None] + list(splits) + [None]

    return [
        data[ (s1 is None or rand >  s1 / norm) & 
              (s2 is None or rand <= s2 / norm) ]
        for s1, s2 in zip(splits[:-1], splits[1:])
    ]


def get_constrained(frame_counts, n=2):
    '''Loosen the constraints of the required frames by clipping using the nth largest constraint.'''
    percent_missing = frame_counts.frames_required / frame_counts.frames_available
    max_n = percent_missing.nlargest(n).iloc[n-1]
    
    frames_required_trunc = frame_counts.frames_required.clip_upper(frame_counts.frames_available * max_n)
    return frames_required_trunc / max_n



'''Misc'''


        
class Exporter(object):

    def __init__(self, name='UOImages', data_dir='Data'):
        self.base_dir = os.path.join(data_dir, name)
        self.image_dir = os.path.join(self.base_dir, 'Images')
        self.ann_dir = os.path.join(self.base_dir, 'Annotations')
        self.split_dir = os.path.join(self.base_dir, 'ImageSets')
        
        # create folders if they don't exist
        ensure_dir(self.image_dir)
        ensure_dir(self.ann_dir)
        for imset in ['Main']:# , 'Layout', 'Segmentation'
            ensure_dir(os.path.join(self.split_dir, imset))
        
    
    def export_imgs_anns(self, pipeline, img_labels, skip_images=False, skip_annotations=False, recompute=False,
                  save_ext='.jpg', bgsubargs=None):
        imdb = pipeline.imdb
        
        if skip_images:
            srcs = pipeline.input_srcs
            srcs = [src for src in srcs if not os.path.join(self.image_dir, pipeline.src_to_idx(src) + save_ext)]
            ims = srcs # fake images
            shape = pipeline.first().shape # need to load one to get the shape
        else:
            ims, srcs = pipeline, pipeline.srcs
            
        for im, src in zip(ims, srcs):
            idx = imdb.src_to_idx(src)
            date = imdb.df.at[src, 'date']
            
            im_path = os.path.join(self.image_dir, idx + save_ext)
            ann_path = os.path.join(self.ann_dir, idx + '.xml')
            
            if not skip_images:
                shape = im.shape
                cv2.imwrite(im_path, im)
                
                if date: # set timestamp of file
                    utils.set_date(im_path, date)
                
            if not skip_annotations and (recompute or not os.path.isfile(ann_path)):
                labels = img_labels.loc[[src]]

                ann_dict = generate_annotation(src, labels, size=shape, 
                                               date=date, 
                                               bgsub_params=bgsubargs)
                root = dict2xml(ann_dict, root='annotation')
                ET.ElementTree(root).write(ann_path)
    

        
    def export_splits(self, df, constraint=False, test_size=0.3):        
        print('generating splits...')
        if constraint:
            idx_subset = resample_by_duration(df, constraint)
        else:
            idx_subset = df.index
        
        print('The resampled dataset contains:')
        print(df.loc[idx_subset].label.value_counts())
        
        print('creating splits...')
        # create splits
        X_test, X_trainval = split_dataset(idx_subset, [test_size])
        X_val, X_train = split_dataset(X_trainval, [test_size])

        splits = [
            ('trainval', X_trainval), 
            ('train', X_train), 
            ('test', X_test),
            ('val', X_val),
            ('all_downsampled', idx_subset), # may be the same as all
            ('all', df.index)
        ]
    
        # get the count of labels in each image
        label_counts = df.groupby(['src', 'label']).w.count().unstack().fillna(0).astype(int)

        print('writing files...')
        section = 'Main'
        for split, idxs in splits:
            files = df.src.loc[idxs].unique()
            
            print(split, 'contains:')
            print(df.loc[idxs].label.value_counts())

            # write the overall splits
            with open(os.path.join(self.split_dir, section, split+'.txt'), 'w') as f:
                for src in files:
                    f.write(filename2index(src))
                    f.write('\n')

            # write files for individual classes
            for label in label_counts.columns:
                label_split = '{}_{}.txt'.format(label, split)
                with open(os.path.join(self.split_dir, section, label_split), 'w') as f:
                    for src in files:
                        contains_label = '1' if label_counts.loc[src].get(label) else '-1'
                        f.write('{} {}\n'.format(filename2index(src), contains_label))
    
    
def resample_by_duration(df, constraint=10):
    only_plumes = df.label == 'plume'
    
    # this is the current distribution of the frame durations. it is heavily biased towards longer plumes.
    frame_distr = df[only_plumes].groupby('plume_dur').src.count()
    # This is the distribution of plume instances. we want our training set to more closely mirror this.
    inst_distr = df[only_plumes].groupby('plume_dur').origin_id.nunique()

    # build frame count distributions
    frame_counts = pd.DataFrame({
        'frames_available': frame_distr, 
        'frames_required': inst_distr / inst_distr.sum() * frame_distr.sum()
    })

    frame_counts['frames_used'] = get_constrained(frame_counts, n=constraint)
    frame_counts['prob'] = (frame_counts.frames_used
                           / frame_counts.frames_used.sum() 
                           / frame_counts.frames_available)


    print('getting probability weightings...')
    plumes_dist = frame_counts.prob[df.loc[only_plumes, 'plume_dur']]
    df.loc[only_plumes, 'prob'] = plumes_dist.values 

    print('resampling dataset...')
    max_dataset_size = int(frame_counts.frames_used.sum())
    print('max resampled dataset size:', max_dataset_size)

    # I changed to this so that we would use all of the negative samples
    plume_idxs = np.random.choice(df[only_plumes].index, size=max_dataset_size, replace=False, 
                                  p=df[only_plumes].prob)
    idx_subset = df[~only_plumes].index.union(plume_idxs)
    return idx_subset
        
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build the UO dataset in Pascal VOC format')
    parser.add_argument('--skip-images', action='store_true',
                    help='skip computing images')
    parser.add_argument('--skip-annotations', action='store_true',
                    help='skip computing annotations')
    parser.add_argument('--skip-splits', action='store_true',
                    help='skip computing splits')
    parser.add_argument('--recompute', action='store_true',
                    help='recompute if existing')
    
    parser.add_argument('-l', '--list', default=None,
                    help='create from list of srcs')
    parser.add_argument('--split-specified', action='store_true',
                    help='dont create splits from all existing files, only those from the list/index')
    parser.add_argument('--entire-day', action='store_true',
                    help='compute for entire day where the day contains labels')
    parser.add_argument('-n', default=None, type=int,
                    help='max number of images to get (for testing or smaller batch)')
    parser.add_argument('--start', default=0, type=int,
                    help='image index offset.')
    parser.add_argument('--label-file', default='labels.csv', 
                        help='the csv of labels to use')
    
    # image generation
    parser.add_argument('--rescale', default=None, type=float,
                    help='proportion to rescale images (dimensions). e.g. use 0.5 to resize to half.')
    parser.add_argument('-w', '--window', default=None, type=int,
                    help='the background subtraction window size')
    parser.add_argument('--blur', default=None, type=int,
                    help="the blur radius. must be odd or you'll get an error. set to 0 to disable")
    parser.add_argument('--scale', default=None, type=float,
                    help='proportion to rescale bgsub images (amplitude). Use when you want to magnify the signal.')
    parser.add_argument('--full-image', action='store_true',
                    help='dont crop the image?')
    
    # training set
    parser.add_argument('--resample-constraint', default=False, type=int,
                    help=("resample the plume distributions to match the distribution of plume instances. "
                          "constraint defines how loose the fitting of the plume distributions are. I was using 10. "
                          "A value of 10 means that the first 9 constraining bins will be ignored. "
                          "leave blank to not resample."))
    parser.add_argument('--test-size', default=0.3, type=float,
                    help="proportion of dataset is for testing")
    
    # output
    parser.add_argument('--save-ext', default='.jpg',
                    help='the filetype to save to')
    parser.add_argument('-d', '--base-dir', default='export',
                    help='base directory where the datasets are stored')
    parser.add_argument('--name', default='UOImages',
                    help='name of the specific dataset')
#       parser.add_argument('--serial', action='store_true',
#                       help='dont use multiprocessing')    
    parser.add_argument('--seed', default=123321, type=int,
                    help='random seed')  
    args = parser.parse_args()
    
    if args.seed:
        np.random.seed(args.seed)
    
    print('I havent tested and debugged this yet..')
    
    exporter = Exporter(args.name, args.base_dir)
    
    # Get the list of files to compute
    print('Getting full list of images...')
    imdb = uoimdb()
    print(imdb.df.shape)
    
    print('loading labels dataset...')
    labels_df = get_ground_truth(args.label_file).set_index('id')
    print(labels_df.shape)
    
    if args.list: # load from a list of file paths
        print('Loading images from file list...')
        with open(args.list, 'r') as f:
            srcs = [src.strip() for src in f]
            
    else: # load all images with labels
        srcs = labels_df.src.unique()
        if args.entire_day:
            print('loading all images on days with labels...')
            dates = imdb.df.loc[srcs].date.dt.date.unique()
            
            print(len(dates), 'dates with labels:', dates)
            srcs = imdb.df[imdb.df.date.dt.date.isin(dates)].index     
    
    # take a subset of images
    if args.start:
        srcs = srcs[args.start:]
    if args.n:
        srcs = srcs[:args.n]
    
    
    # TODO: need to store register of all pipe operations in annotations, or somewhere where we can reconstitute them.
    # get bg subtraction params
    bgsubargs = dict(crop=not args.full_image, window_size=args.window, blur_radius=args.blur, scale=args.scale)
    bgsubargs = {k: v for k, v in bgsubargs.items() if v is not None}
    
    # set scaling parameter (changes the dimensions of the image)
    if args.rescale:
        imdb.cfg.RESCALE = args.rescale
        
    if args.window:
        imdb.cfg.BG.WINDOW = [args.window]*2

    if args.blur:
        imdb.BG.BLUR_RADIUS = args.blur

    if args.scale:
        args.BG.SCALE = args.scale
        
    # get the labels grouped by each image
    img_labels = labels_df.reset_index()
    img_labels = img_labels[img_labels.src.isin(srcs)].set_index('src')


    ## build pipeline

    # feed images
    pipeline_single = imdb.pipeline().use_window()
    if not args.full_image:
        pipeline_single.crop()
    pipeline_single.single_bgsub2() # will yield a single background sub image each time it's fed.
    
    pipeline = imdb.feed_images((pipeline_single.feed(src=src).first() for src in srcs), srcs=srcs).progress(1)
    

    if not args.skip_images or not args.skip_annotations:
        print('Beginning background subtraction and annotation export...')
        print( '{} images & {} annotations existing'.format(
            len(glob.glob(os.path.join(exporter.image_dir, '*' + args.save_ext))),
            len(glob.glob(os.path.join(exporter.ann_dir, '*.xml'))) ))
        
        exporter.export_imgs_anns(pipeline, img_labels, skip_images=args.skip_images, skip_annotations=args.skip_annotations) # , bgsubargs=bgsubargs
        
    
    if args.list or args.split_specified:
        loaded_srcs = srcs
    else: # split all existing. just in case anything failed.
        loaded_srcs = [
            os.path.splitext(os.path.basename(p).replace(',', '/'))[0]
            for p in glob.glob(os.path.join(exporter.image_dir, '*'))
        ]
    
    # remove images that failed to load
    labels_df1 = labels_df[labels_df.src.isin(loaded_srcs)].copy()
    print(labels_df1.shape, len(loaded_srcs))
    
    '''Generate splits'''
    
    if not args.skip_splits:
        exporter.export_splits(labels_df1, constraint=args.resample_constraint, test_size=args.test_size)
        
            
    print('------------------------------------------------------------')
    print('`{}` Dataset Generation Finished.'.format(args.name))
    
    
    
    
