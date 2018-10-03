import os
import cv2
import glob
import time
import hashlib
import dill as pickle
from datetime import datetime
from collections import deque
from functools import partial, reduce
from itertools import chain

import imageio
import numpy as np
import pandas as pd
import matplotlib as mpl
import multiprocessing as mp
import matplotlib.cm

from . import utils
from .utils import null_image
from .dataset import transform_boxes_crop
from .config import get_config


class uoimdb(object):
    '''Build end-to-end image pipelines using the UO images.
    All images and their timestamps are loaded into a dataframe, self.df.
    
    Definitions:
        src: refers to the relative filename (relative to cfg.IMAGE_BASE_DIR)
        idx: refers to the unique index that can be used as a filename (derived from src)
        path: refers to the absolute path of the image
    
    Initialization:
        imdb = uoimdb()
        imdb.df.head()
    
    Build pipelines as follows:
        srcs = imdb.df.index[:10] # take the first 10 image paths
        ims = imdb.load_images(srcs).crop().bgsub2() # pipeline is created when calling load_images(...)
        for im, src in zip(ims, ims.srcs): # src must come after ims
            plt.imshow(im)
            plt.title(src)
            plt.show()
        
    Export as a gif:
        from IPython.display import Image
        Image(url=imdb.load_images(srcs)
                      .crop().bgsub2()
                      .to_gif('my-fancy-gif'))


    Various ways to build pipelines:
        imdb.load_images(imdb.index[:10]) # loading images from a list of srcs
        imdb.feed_images(iterator_of_images_data) # creating a pipeline from already loaded images
        imdb.load_around(src, window=(5,5)) # load 5 images on either side of src

        # These are all aliases for calling this
        imdb.pipeline().feed(
            imgs=imgs, # you can specify a series of images to pass in
            srcs=srcs, # or more commonly, a list of srcs
            src=src, window=(7,5), # or a single src with optionally images on either side
        )
    
    '''
        
    def __init__(self, cfg=None, df=None, refresh=False, pipeline_init=True, **kw):
        # load the configuration options and add in any overrides
        self.cfg = get_config(cfg)
        self.cfg.update(**kw)

        # setup the loaded image buffer. keeps memory usage from blowing up
        self.image_cache_list = deque(maxlen=self.cfg.CACHE_SIZE) # stores images in load order to limit memory consumption

        # setup the pipeline init function. what is called everytime a pipeline is created. Useful to avoid having to run bgr2rgb everytime.
        self.pipeline_init = None
        if pipeline_init: # can set to false to disable
            if callable(pipeline_init): # use passed in init function
                self.pipeline_init = pipeline_init
            elif self.cfg.PIPELINE_INIT: # can disable via config file
                if self.cfg.RESCALE and self.cfg.RESCALE != 1:
                    self.pipeline_init = lambda pipeline: pipeline.bgr2rgb().resize(self.cfg.RESCALE)
                else: # rescale is disabled
                    self.pipeline_init = lambda pipeline: pipeline.bgr2rgb()
            
        # get the image db cache location
        self._cache_id = hashlib.md5(os.path.abspath(self.abs_file_pattern).encode('utf-8')).hexdigest()
        self._cache_file = os.path.join(self.cfg.DATA_LOCATION, 'imdb-{}.pkl'.format(self._cache_id))

        if not os.path.isdir(self.cfg.DATA_LOCATION):
            os.makedirs(self.cfg.DATA_LOCATION)

        # load the image list
        if df is not None: # use preloaded df
            self.df = df

        elif not refresh and os.path.isfile(self._cache_file): # load from file
            self.df = pd.read_pickle(self._cache_file)

        else: # create imdb cache
            print(('Missing {}. Needs to extract file timestamps which could take a WHILE. For >1M images, it was taking 20-30 mins. '
                   'This only needs to run once though and subsequent times will load the cached pickle file and '
                   'finish in seconds.').format(self._cache_file))
        
            self.df = pd.DataFrame([
                dict(
                    src=self.path_to_src(path),
                    date=self.get_path_date(path)) 

                for path in self.load_image_files()

            ], columns=['src', 'date', 'im']).set_index('src').sort_values('date')

            self.df.im = self.df.im.astype(object)
            self.df['idx'] = self.df.index.map(self.src_to_idx)
            
            if len(self.df):
                self.df.to_pickle(self._cache_file)


            
    '''Filename handling functions'''

    @property
    def abs_file_pattern(self):
        return os.path.join(self.cfg.IMAGE_BASE_DIR, self.cfg.IMAGE_FILE_PATTERN)

    def load_image_files(self):
        return glob.glob(self.abs_file_pattern)
        
    def idx_to_src(self, idx):
        return idx.replace(',', '/')

    def src_to_idx(self, src):
        return src.replace('/', ',')

    def idx_to_path(self, idx):
        return os.path.join(self.cfg.IMAGE_BASE_DIR, self.idx_to_src(idx))

    def src_to_path(self, src):
        return os.path.join(self.cfg.IMAGE_BASE_DIR, src)
    
    def path_to_src(self, src):
        return os.path.relpath(src, self.cfg.IMAGE_BASE_DIR)

    def src_to_i(self, src):
        return self.df.index.get_loc(src) if src in self.df.index else None

    def idx_to_i(self, idx):
        return self.src_to_i(self.idx_to_src(idx))

    def get_path_date(self, path):
        '''Get the timestamp from an image path'''
        return datetime.fromtimestamp(os.stat(path).st_mtime)
    
    
    '''Building pipeline'''

    def pipeline(self, raw=False):
        pipeline = Pipeline(self)
        if not raw and self.pipeline_init:
            self.pipeline_init(pipeline)
        return pipeline
        
    def load_around(self, src, window=None, raw=False, **kw):
        '''Initializes the image pipeline with images surrounding a specific file.'''
        if window is None:
            window = cfg.BG.WINDOW

        pipeline = Pipeline(self).feed(src=src, window=window, **kw)
        if not raw and self.pipeline_init:
            self.pipeline_init(pipeline)
        return pipeline
        
    def load_images(self, srcs, raw=False, **kw):
        '''Initializes the image pipeline with a list of files to load'''
        pipeline = Pipeline(self).feed(srcs, **kw)
        if not raw and self.pipeline_init:
            self.pipeline_init(pipeline)
        return pipeline

    def feed_images(self, imgs, **kw):
        '''Initializes the image pipeline with a list of already loaded images'''
        return Pipeline(self).feed(imgs=imgs, **kw)
  
    def load_image(self, src, raw=False, **kw):
        '''Initialize the image pipeline with a single image'''
        pipeline = Pipeline(self).feed(src=src, **kw)
        if not raw and self.pipeline_init:
            self.pipeline_init(pipeline)
        return pipeline


    '''Image retrieval'''

    def load_image_data(self, src):
        return cv2.imread(self.src_to_path(src))

    def get_image(self, src, refresh=False):
        '''Retrieve an image by src.'''
        im = self.df.at[src, 'im']
        if refresh or null_image(im):
            im = self.df.at[src, 'im'] = self.load_image_data(src)
        return im
    

    def around_i(self, i, window=None):
        if window is None:
            window = cfg.BG.WINDOW
        n_prev, n_after = (window,window) if isinstance(window, int) else window
        return self.df.index[max(i-(n_prev or 0), 0):i+(n_after or 0)+1]

    def around_src(self, src, window=None):
        if window is None:
            window = cfg.BG.WINDOW
        i = self.src_to_i(src)
        return self.around_i(i, window) if i is not None else []

    def around_idx(self, idx, window=None):
        if window is None:
            window = cfg.BG.WINDOW
        i = self.idx_to_i(idx)
        return self.around_i(i, window) if i is not None else []


    

    def _load_images(self, srcs, pool=None, out_srcs=None, close=None, refresh=False):
        if pool is not None:
            srcs = pd.Series(srcs)
            missing = (pd.isnull(self.df.loc[srcs, 'im']).reset_index(drop=True) 
                       if not refresh else np.ones(len(srcs)).astype(bool))
            missing_files = srcs[missing].apply(self.src_to_path)
            
            if len(missing_files):
                if pool is True:
                    pool = len(srcs) # default 1 process per file
                    
                if close is None: # whether or not to close the pool
                    close = isinstance(pool, int) # close if we created it
                    
                if isinstance(pool, int): # pool is the number of processes
                    pool = mp.Pool(min(pool, self.cfg.MAX_PROCESSES)) # mp.Pool(pool) 

                print('{} images total. {} missing. {} workers. {} cpus total.'.format(
                    len(srcs), len(missing_files), pool._processes, mp.cpu_count()))
                
                queue = pool.imap(load_image_worker, missing_files, chunksize=5)
                for im in self._iter_images(srcs, queue=queue, out_srcs=out_srcs, refresh=refresh):
                    yield im
                 
                if close:
                    pool.close()
                    pool.join()
            else: # all images loaded. no need to open a pool
                for im in self._iter_images(srcs, out_srcs=out_srcs, refresh=refresh):
                    yield im
        else: # load serially
            for im in self._iter_images(srcs, out_srcs=out_srcs, refresh=refresh):
                yield im
        
        
    def _iter_images(self, srcs, queue=None, out_srcs=None, refresh=False):
        '''Load images from file.
        Arguments:
            srcs (list-like): the subset of images (src) to load
        '''
        
        # loop over files in order
        for src in srcs:
            im = self.df.at[src, 'im']

            if refresh or null_image(im):
                if queue is not None: # load from multiprocessing queue
                    while null_image(im):
                        qsrc, qimg = next(queue)
                        qsrc = self.path_to_src(qsrc)
                        if qimg is None and qsrc == src: 
                                break
                        
                        self.add_image(qsrc, qimg)
                        im = self.df.at[src, 'im'] # get updated value
            
                else: # load normally
                    im = self.load_image_data(src)
                    self.add_image(src, im)
            
            if im is None:
                print('could not load image: {}'.format(src))
                continue
            
            # only save the filenames for the images that have actually been loaded
            if out_srcs is not None:
                out_srcs.append(src) 
            yield im
    
    
    def add_image(self, src, im):
        '''Add image data to pandas table'''
        self.df.at[src, 'im'] = im # add image
        
        # if queue is full, remove image for the last one
        if len(self.image_cache_list) == self.image_cache_list.maxlen:
            self.df.at[self.image_cache_list[0], 'im'] = None
        self.image_cache_list.append(src) # add to queue
        return self
    
    
    def clear_images(self):
        '''Clear all image data'''
        self.df.loc[:, 'im'] = None
        self.image_cache_list.clear()
        return self
    

    
    
    
    

    
    
class Pipeline(object):
    '''Image processing pipelines.
    Add functions to the pipeline by chaining. 
    Iterate over pipeline to get the resultant images.
    Note: The pipline is baked before iterating so adding pipes 
          after iteration has started will do nothing.
    
    You don't need to create this class directly, it will be generated using:
    imdb = uoimdb()
    pipe = imdb.load_images(imdb.df.index[:10])
    
    Available Pipes:
        color_convert: convert the color from one color scale to another
        grey: convert to greyscale
        bgr2rgb: convert from bgr2rgb
        
        crop: crop the image.
        resize: resize the image dimensions
        astype: convert the image dtype
        
        blur: blur the image
        mean_shift: subtract off the image mean
        scale: scale the image values. useful for low signal
        
        bgsub: perform basic background subtraction
        bgsub2: perform background subtraction with all of the bells and whistles.
        
        clip: clip between 0 and 255
        norm: normalize between 0 and 255 (or specified bounds)
        invert: subtract image from 255
        cmap: apply a colormap. warning, this returns a float.
        
        progress: print out the iteration number at a set interval.
        
        Add a custom function using:
            # custom function. runs once per image
            imdb.load_images(srcs).pipe(custom_func, *a, **kw)
            # or custom generator. takes in the pipeline up to this point
            imdb.load_images(srcs).pipe(custom_gen, full=True, *a, **kw)
        Just make sure that the image data is the first and only required argument 
            after binding *a and **kw using partial(...)
            
        
    Export Options:
        save_images: save to image file. default jpg
        save_gif: export to gif
        
        Or, it's easy enough to just do it yourself:
            for im in imdb.load_images(srcs).grey():
                plt.imshow(im)
                plt.show()
          
    '''
    
    def __init__(self, imdb, **kw):
        self.srcs = [] # the loaded filenames
        self.pipeline = [] # the pipeline of generators
        self.box_transforms = [] # transforms to apply to bounding boxes
        self.imdb = imdb # stores all of the images
        self.current_i = None # the current loop iteration
        self._window = None # the number of frames to load around a singular src
        self.clear_feed() # initialize variables


    @property
    def current_src(self):
        return self.srcs[self.current_i] if self.current_i is not None and len(self.srcs) else ''
    
    @property
    def current_idx(self):
        return self.imdb.src_to_idx(self.current_src)

    @property
    def cfg(self):
        return self.imdb.cfg

            
    def feed(self, srcs=None, imgs=None, src=None, window=None, **kw):
        '''Define the pipeline input. Pass either srcs, src, or imgs.
        Arguments:
            srcs (iter of str): list of image srcs to use
            imgs (iter of imgs): list of images (np.array) to use
            src (str): the src of the foreground image
            window (int or tuple): the number of images on either side (used only with src). 
                                    int for equal, tuple to specify count on each side.
            **kw: any additional keyword args to pass to the loader.
        '''
        if srcs is None and imgs is None and src is None:
            srcs = self.input_srcs # reload the image generator
        
        self._iter = None # stores the pipeline iterator
        self.input_srcs = srcs
        self.i_fg = None
        self.src_fg = None
        self.srcs = []
        window = window if window is not None else self._window

        if src is not None: # get images around src
            self.src_fg = src
            if window is not None:
                srcs = self.imdb.around_src(src, window=window) # get srcs around the fg image
                self.i_fg = srcs.get_loc(src) # store the location of foreground
            else:
                srcs = (src,)
            
        if imgs is not None: # use list of images
            self._input = imgs
            self.srcs = srcs
        elif srcs is not None: # load list of srcs
            self._input = self.imdb._load_images(srcs, out_srcs=self.srcs, **kw)
        return self 


    def clear_feed(self):
        '''Clear the feed variables'''
        self.input_srcs = []
        self.i_fg = None
        self.src_fg = None
        self.srcs = []
        self._input = None
        self._iter = None


    def use_window(self, window=None):
        '''set the window to be used when a single src is passed in.'''
        if window is None:
            window = self.cfg.BG.WINDOW
        self._window = window

        if self.src_fg:
            self.feed(self.src_fg)
        return self

    
    def pipe(self, func, full=False, box_transform=None, singleton=False, *a, **kw):
        '''Add a function/generator to the pipeline. 
        Arguments:
            func (callable): the function to add to the image pipeline.
                Image must be first argument after calling partial()
            full (bool): If True, it passes the full iterable to the function instead of each element.
            box_transform (callable): a function that is used to transform bounding boxes e.g. when cropping or resizing images.
                Must take a dataframe of bounding boxes as the only required argument.
        '''

        f = partial(func, *a, **kw)
        if full:
            if singleton:
                self.pipeline.append(lambda *a,**kw: (f(*a,**kw),))
            else:
                self.pipeline.append(f)
        else:
            self.pipeline.append(lambda ims: (f(im) for im in ims))
        if box_transform is not None:
            self.box_transforms.append(box_transform)
        return self
    
    
    def __call__(self, *a, **kw):
        '''Compile the pipeline.'''
        if len(a) or len(kw):
            self.feed(*a, **kw)

        if self._input is not None: # load images one after another
            # generate image pipeline
            pipeline = self._input
            for func in self.pipeline:
                pipeline = func(pipeline)
        else:
            pipeline = ()
        return pipeline

    def run(self):
        '''Alias for ppl uncomfortable with double calling :p i.e. func()().'''
        return self()

    def __iter__(self):
        '''Iterate over the processed images by iterating over the pipeline'''
        for i, im in enumerate(self()):
            self.current_i = i
            yield im
        self.current_i = None

    def __next__(self):
        '''Now pipelines are officially iterators.'''
        if not self._iter:
            self._iter = iter(self)
        return next(self._iter)

    def first(self):
        '''Get the first element. Useful for single element iterators'''
        return next(self) # keep 

    def borrow_first(self):
        a = self.first()
        self._iter = chain((a,), self._iter) # tack it back onto the beginning
        return a

    def tolist(self):
        '''Convert iterator to list'''
        return list(self)

    #def clone(self):
    #    '''Copy the image pipeline. Warning: this doesn't change the references to the pipeline in the pipe closures, so basically this won't work.'''
    #    pipeline = Pipeline(self.imdb)
    #    pipeline.pipeline = [f for f in self.pipeline]
    #    pipeline.box_transforms = [f for f in self.box_transforms]
    #    return pipeline


    def save(self, path):
        '''Save pipeline to file'''
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return self

    @staticmethod
    def load(path):
        '''Load pipeline from file'''
        with open(path, 'rb') as f:
            return pickle.load(f)

    '''Pipes'''

    def init(self):
        '''Run the pipeline initialization function if it exists.'''
        if self.imdb.pipeline_init:
            self.imdb.pipeline_init(self)
        return self

   
    def crop(self, x1=None, x2=None, y1=None, y2=None):
        '''Crop the images. 
        Arguments:
            x1, x2, y1, y2 (float): The percentage of the image to crop off. The full image would be:
                x1, x2, y1, y2 = 0, 0, 1, 1
        '''
        if x1 is None and x2 is None and y1 is None and y2 is None:
            x1, y1, x2, y2 = self.cfg.CROP.X1, self.cfg.CROP.Y1, self.cfg.CROP.X2, self.cfg.CROP.Y2

        x1, y1, x2, y2 = x1 or 0, y1 or 0, x2 or 1, y2 or 1
        box_transform = partial(transform_boxes_crop, out_crop=((x1, y1), (x2, y2)))

        def f(im):
            h, w = im.shape[:2]
            crop_x = slice(int(w*x1), int(w*x2))
            crop_y = slice(int(h*y1), int(h*y2))
            return im[crop_y, crop_x]
        return self.pipe(f, box_transform=box_transform)
    

    def color_convert(self, color_conv):
        '''Convert the color space of the images. 
        Arguments:
            color_conv: An opencv color space. e.g. cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2RGB, etc.
        Returns:
            Pipe returns iterable of converted images.
        '''
        return self.pipe(lambda im: cv2.cvtColor(im, color_conv) if len(im.shape) > 2 else im)
    

    def grey(self, color_conv=cv2.COLOR_BGR2GRAY):
        '''Convert images to greyscale. 
        Arguments:
            color_conv: An opencv color space. e.g. cv2.COLOR_BGR2GRAY, cv2.COLOR_RGB2GRAY, cv2.COLOR_HSV2GRAY, etc.
        Returns:
            Pipe returns iterable of converted images.
        '''
        return self.color_convert(color_conv=color_conv)
    

    def bgr2rgb(self):
        '''Convert images from BGR (opencv default) to RGB (normal people default). 
        Returns:
            Pipe returns iterable of converted images.
        '''
        return self.color_convert(color_conv=cv2.COLOR_BGR2RGB)


    def rgb2bgr(self):
        '''Convert images from RGB to BGR, typically for image saving. 
        Returns:
            Pipe returns iterable of converted images.
        '''
        return self.color_convert(color_conv=cv2.COLOR_RGB2BGR)
    

    def resize(self, scale=None):
        '''Resize the images by some proportional scaling parameter. 
        Arguments:
            scale (float): The proportional scaling parameter. e.g. scale in half using scale=0.5 
        Returns:
            Pipe returns iterable of resized images.
        '''
        if scale is None:
            scale = self.cfg.RESCALE
        return self.pipe(lambda im: cv2.resize(im, None, fx=scale, fy=scale))
    

    def astype(self, dtype):
        '''Convert the image data type. 
        Arguments:
            dtype (str, type): Anything you would use with np.array([]).astype(...)
        Returns:
            Pipe returns iterable of converted images.
        '''
        return self.pipe(lambda im: im.astype(dtype))
    

    def bgsub(self, window=None):
        '''Performs basic background subtraction. 
        Arguments:
            window (int): The size of the background subtraction window. Window is equal on each side (left-heavy for even-sized windows).
        Returns:
            Pipe returns iterable of background subtracted images.
        '''
        if window is None:
            window = self.cfg.BG.WINDOW
        return self.pipe(consecutive_bgsub, full=True, window=window)
    

    def bgsub2(self, window=None, cmap=None):
        '''Performs fancy background subtraction. Includes the full pipeline:
            convert to greyscale
            blur for translation invariance
            ** background subtraction
            blur for further noise removal
            scale up to increase visibility of differences
            clip between 0 and 255 to avoid overflow
            invert so white is static and black represents motion
            convert to 'bone' cmap so it's prettier
            convert to uint8 so matplotlib is happy

        Arguments:
            window (int): The size of the background subtraction window. Window is equal on each side (left-heavy for even-sized windows).
        Returns:
            Pipe returns iterable of background subtracted images.
        '''
        if window is None:
            window = self.cfg.BG.WINDOW
        if cmap is None:
            cmap = self.cfg.BG.CMAP

        (self.grey()
             .blur()
             .bgsub(window=window)
             .blur()
             .scale()
             .clip()
             .invert()
             .cmap(cmap)
             .astype('uint8'))
        return self
    

    def single_bgsub(self, method='mean'):
        '''Performs basic background subtraction. 
        Arguments:
            window (int): The size of the background subtraction window. Window is equal on each side (left-heavy for even-sized windows).
        Returns:
            Pipe returns a single background subtracted image.
        '''
        def f(ims):
            ims = list(ims)
            i = self.i_fg if self.i_fg is not None else (len(ims) + 1) // 2
            im, bgims = ims[i], ims[:i] + ims[i+1:]
            return disjoint_bgsub(im, bgims, method=method)
        return self.pipe(f, full=True, singleton=True)

    def single_bgsub2(self, method='mean'):
        '''Performs fancy background subtraction for a single image. Includes the full pipeline:
            convert to greyscale
            blur for translation invariance
            ** background subtraction
            blur for further noise removal
            scale up to increase visibility of differences
            clip between 0 and 255 to avoid overflow
            invert so white is static and black represents motion
            convert to 'bone' cmap so it's prettier
            convert to uint8 so matplotlib is happy

        Arguments:
            window (int): The size of the background subtraction window. Window is equal on each side (left-heavy for even-sized windows).
        Returns:
            Pipe returns iterable of background subtracted images.
        '''
        (self.grey()
             .blur()
             .single_bgsub(method=method)
             .blur()
             .scale()
             .clip()
             .invert()
             .cmap(self.cfg.BG.CMAP)
             .astype('uint8'))
        return self
    

    def blur(self, blur_radius=None):
        '''Blurs each image with a specified blur radius. 
        Arguments:
            blur_radius (int, odd): The blur radius.
        Returns:
            Pipe returns iterable of blurred images.
        '''
        if blur_radius is None:
            blur_radius = self.cfg.BG.BLUR_RADIUS
        blur_radius = blur_radius + (1 - blur_radius%2) # BLUR HAS TO BE ODD

        return self.pipe(lambda im: cv2.GaussianBlur(im, (blur_radius,blur_radius), 0))
    

    def mean_shift(self):
        '''Subtracts out the average value across the image. 
        Returns:
            Pipe returns iterable of modified images.
        '''
        return self.pipe(lambda im: im - im.mean())
    

    def scale(self, scale=None):
        '''Multiply the image by some scaling factor.
        Arguments:
            scale (float): The scale. To double image pixel values, set scale=2
        Returns:
            Pipe returns iterable of scaled images.
        '''
        if scale is None:
            scale = self.cfg.BG.SCALE

        return self.pipe(lambda im: im*scale)
    

    def clip(self, vmin=0, vmax=255):
        '''Clip values.
        Arguments:
            vmin (float): the minimum value. Default is 0
            vmax (float): the maximum value. Default is 255
        Returns:
            Pipe returns iterable of clipped images.
        '''
        return self.pipe(lambda im: np.clip(im, vmin, vmax))
    

    def norm(self, vmin=0, vmax=255):
        '''Normalize values between two values.
        Arguments:
            vmin (float): the minimum value. Default is 0
            vmax (float): the maximum value. Default is 255
        Returns:
            Pipe returns iterable of normalized (between 0,1) images.
        '''
        return self.pipe(lambda im: (im - im.min()) / (im.max() - im.min()) * (vmax - vmin) + vmin)


    def norm_std(self, nstd=1):
        '''Normalize values by mean and std deviation.
        Returns:
            Pipe returns iterable of normalized images.
        '''
        return self.pipe(lambda im: (im - im.mean()) / (nstd * (im.std() or 1)))
    

    def invert(self, vmax=255):
        '''Invert the image (0 becomes 255, 255 becomes 0).
        Arguments:
            vmax (float): the maximum value to subtract from. Default is 255
        Returns:
            Pipe returns iterable of inverted images.
        '''
        return self.pipe(lambda im: vmax - im)
    

    def cmap(self, cmap='bone'):
        '''Apply a colormap to the image.
        Arguments:
            cmap (str or matplotlib colormap): the matplotlib colormap to use.
        Returns:
            Pipe returns iterable of images with colormap applied.
        '''
        if isinstance(cmap, str):
            cmap = mpl.cm.get_cmap(cmap)
        
        return self.pipe(lambda im: cmap(im/255.)*255)


    def draw_detections(self, detections, **kw):
        '''Draws detections on the images.
        Arguments:
            detections (pd.DataFrame): dataframe of detections. Index must be the image src and must contain
                the columns: x1, y1, x2, y2 which represent the box coordinates on the uncropped image between 0 and 1.
        Returns:
            Pipe returns iterable of images with detections drawn.
        '''
        box_transform = lambda boxes: reduce(lambda boxes, f: f(boxes), self.box_transforms, boxes)
        def f(im, **kw):
            if im.idx in detections.index:
                b = detections.loc[[im.idx]]
                boxes = box_transform(b)
                return draw_dets_cv(im, boxes, **kw) 
            else:
                return im
        return self.attach_meta().pipe(f, **kw)

    def attach_meta(self, **kw):
        '''Bind the image src and other meta to the numpy array. Needs to be called intermittenly because 
            the meta would be lost if a copy was returned by any pipes.
        Returns:
            Pipe returns iterable of images with src, idx, and i added as attributes to the image np.array.
        '''
        def f(ims):
            for i, im in enumerate(ims):
                src = self.srcs[i]
                idx = self.imdb.src_to_idx(src)
                yield im.view(utils.metaArray).set_meta(i=i, src=src, idx=idx, **kw)
        return self.pipe(f, full=True)

    def timer(self, every=1):
        '''Prints execution time'''
        def timer(imgs):
            t = time.time()
            buf = []
            for i, _ in enumerate(imgs):
                if every and i and not i % every:
                    print('Time since last: {}. Avg iteration time: {}+-{} (2std). Total time: {}.'.format(np.sum(buf[-every]), np.mean(buf[-every:]), np.std(buf[-every:]), np.sum(buf)))
                yield _
                buff.append(time.time() - t)
                t = time.time()
            print('Total Time: {}. Average Time: {}+-{} (2std).'.format(np.sum(buf), np.mean(buf), np.std(buf)))

        return self.pipe(timer, full=True)


    def progress(self, every=1):
        '''Prints the iteration number.
        Arguments:
            every (int): after how many iterations to update. e.g. print out every=100 iterations
        Returns:
            Pipe returns iterable of the input images.
        '''
        return self.pipe(utils.progress, full=True, every=every)


    '''Export images. Nothing can be piped after these'''
    

    def save_images(self, out_dir=None, ext=None):
        '''Saves the image as an image file. Yields the path that it was saved to.
        Arguments:
            out_dir (str): the output directory. Defaults to imdb.cfg.IMAGE_OUTPUT_DIR.
            ext (str): the file extension of the image. See cv2.imwrite for allowed formats. Defaults to imdb.SAVE_EXT.
        Returns:
            Pipe returns iterable of image paths. Timestamp added to query string to prevent caching.
        '''
        out_dir = out_dir or self.imdb.cfg.IMAGE_OUTPUT_DIR
            
        if not os.path.exists(out_dir): # make sure the save directory exists
            os.makedirs(out_dir)

        def f(ims):
            for im, src in zip(ims, self.srcs):
                path = os.path.join(out_dir or self.imdb.cfg.IMAGE_OUTPUT_DIR, self.imdb.src_to_idx(src) + (ext or self.imdb.cfg.SAVE_EXT))
                cv2.imwrite(path, im)
                yield '{}?{}'.format(path, time.time()) # prevent caching :D suck it chrome.
        return self.pipe(f, full=True)
    

    def save_gif(self, name, out_dir=None, duration=0.3, **kw):
        '''Saves the image as a gif. Returns the path that it was saved to.
        Arguments:
            name (str): the name of the gif, used for the filename.
            out_dir (str): the output directory. Defaults to imdb.cfg.IMAGE_OUTPUT_DIR.
            duration (float): the number of seconds per frame. Default is 0.3s.
        Returns:
            Pipe returns path to gif. Timestamp added to query string to prevent caching.
        '''
        out_dir = out_dir or self.imdb.cfg.IMAGE_OUTPUT_DIR
        gif_file = os.path.join(out_dir, name+'.gif')
        
        if not os.path.exists(out_dir): # make sure the save directory exists
            os.makedirs(out_dir)

        def f(ims): 
            with imageio.get_writer(gif_file, mode='I', duration=duration, **kw) as writer:
                for im in ims:
                    writer.append_data(im)
            return '{}?{}'.format(gif_file, time.time()) # prevent caching :D suck it chrome.
        return self.pipe(f, full=True)
    







def load_image_worker(path):
    '''Multiprocessing worker for loading images'''
    try:
        return path, cv2.imread(path)
    except KeyboardInterrupt:
        raise Exception
    return


def disjoint_bgsub(img, imgs, method='mean'):
    '''Performs one-off background subtraction given the foreground image and background images. 
    Can use alternate subtraction methods'''
    if method == 'mean':
        bg = np.mean(imgs, axis=0)
    elif method == 'hamming':
        window = np.hamming(len(imgs))
        window /= window.sum() # norm
        # weighted average over all images
        bg = (imgs * window[:,None,None]).sum(axis=0)
    elif method == 'median':
        bg = np.median(imgs, axis=0)
    elif method == 'min':
        bg = np.min(imgs, axis=0)
    elif method == 'max':
        bg = np.max(imgs, axis=0)
    else: # == 'mean'
        bg = np.mean(imgs, axis=0)
    
    sub = np.abs(img - bg)
    sub = np.clip(sub, 0, 255)
    return sub

    
def consecutive_bgsub(frames, window):
    '''Performs an optimized version of background subtraction where it is assumed that the images are consecutive.'''
    center, window_right = window # _____,_,_____
    center += 1 # shift one to include the center frame. 
    frames = (frame.astype(float) for frame in frames) # convert to float or everything goes to shit *-*
    buffer = utils.npBuffer([frame for _, frame in zip(range(center + window_right), frames)]) # initialize the buffer
    
    for i in range(center): # start at left edge, i = 0 -> center
        yield np.abs(buffer[i] - buffer.mean_)
    
    for new_frame in frames: # i == center
        buffer.append(new_frame) # store new image
        yield np.abs(buffer[i] - buffer.mean_)
    
    for i in range(center, len(buffer)): # we've hit the right edge, finish up. i = center -> window_size - 1
        yield np.abs(buffer[i] - buffer.mean_)


def draw_dets_cv(im, boxes=None, text_color=(0,0,0)):
    H, W = im.shape[:2]
    im = np.array(im)
    for id, box in boxes.iterrows():
        xs = sorted((int(box.x1*W), int(box.x2*W)))
        ys = sorted((int(box.y1*H), int(box.y2*H)))
        pt1, pt2 = zip(xs, ys)
        # print(pt1, pt2)
        if 'score' in box:
            label = '{} {:.3f}'.format(box.label, box.score)
        else:
            label = box.label

        cv2.rectangle(im, pt1, pt2, (0,0,255), 2)
        cv2.putText(im, label, pt1, cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color, 2)
    return im



