import os
import cv2
import matplotlib as mpl
from collections import OrderedDict as odict


class ImageProcessor(object):

    def __init__(self, imdb):
        self.imdb = imdb

        # all filters
        self.filters = odict([
            ('Background Subtraction (mean)', imdb.pipeline().use_window().single_bgsub3(method='mean')),
            ('Background Subtraction (median)', imdb.pipeline().use_window().single_bgsub3(method='median')),
            ('Background Subtraction (min)', imdb.pipeline().use_window().single_bgsub3(method='min')),
            ('Background Subtraction (max)', imdb.pipeline().use_window().single_bgsub3(method='mean')),

            ('Original', imdb.pipeline()),
            ('Greyscale', imdb.pipeline().grey()),
            ('Edges', imdb.pipeline().grey().pipe(lambda im: cv2.Laplacian(im,cv2.CV_64F)).invert()),

            # https://www.learnopencv.com/non-photorealistic-rendering-using-opencv-python-c/
            ('Stylization', imdb.pipeline().pipe(lambda im: cv2.stylization(im, sigma_s=10, sigma_r=0.4))),
            ('Pencil Sketch', imdb.pipeline().pipe(lambda im: cv2.pencilSketch(im, sigma_s=10, sigma_r=0.1, shade_factor=0.02)[1])),
            ('Detail Enhance', imdb.pipeline().pipe(lambda im: cv2.detailEnhance(im, sigma_s=20, sigma_r=0.15))),
            ('Edge Preserving', imdb.pipeline().pipe(lambda im: cv2.edgePreservingFilter(im, flags=1, sigma_s=30, sigma_r=0.4))),
        ])

        for name in self.filters:
            self.filters[name].fake_crop()


    def process_image(self, filename, filter=None, **kw):
        filter = filter or 'Original'
        if filter not in self.filters:
            return None

        # get image
        img = self.filters[filter].feed(src=filename, **kw).first()

        # convert from rgb2bgr for opencv
        if len(img.shape) < 3:
            pass
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        return img


    def pre_render_processing(self, img, cmap=None, downsample=None):
        '''Runs image processing that runs on both on-the-fly images and cached images'''
        if len(img.shape) < 3:
            if cmap is None or cmap is True:
                cmap = self.imdb.cfg.ON_RENDER_CMAP
            if cmap is True:
                cmap = self.imdb.cfg.BG.CMAP
            if cmap:
                cmap = mpl.cm.get_cmap(cmap)
                img = (cmap(img/255.)*255).astype('uint8')
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        if downsample is None or downsample is True:
            downsample = self.imdb.cfg.ON_RENDER_DOWNSAMPLE
        if downsample is True:
            downsample = self.imdb.cfg.DOWNSAMPLE
        if downsample:
            scale = self.imdb.cfg.ON_RENDER_DOWNSAMPLE
            img = img[::scale, ::scale]

        return img


    def cache_filename(self, filename, filter, ext=None):
        ext = ext or self.imdb.cfg.IMAGE_CACHE_EXT
        if ext:
            filename = os.path.splitext(filename)[0] + '.{}'.format(ext)
        filename = '{},{}'.format(filter.replace('/', ','), self.imdb.src_to_idx(filename))
        return os.path.realpath(os.path.join(self.imdb.cfg.IMAGE_CACHE_LOCATION, filename))

