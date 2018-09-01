import cv2
from collections import OrderedDict as odict


class ImageProcessor(object):
	def __init__(self, imdb):
		self.imdb = imdb

		# background subtraction filters
		bg_filters = [
			('Background Subtraction (mean)', imdb.pipeline().single_bgsub2(method='mean')),
			('Background Subtraction (median)', imdb.pipeline().single_bgsub2(method='median')),
			('Background Subtraction (min)', imdb.pipeline().single_bgsub2(method='min')),
			('Background Subtraction (max)', imdb.pipeline().single_bgsub2(method='mean')),
		]

		# used to know which filters need surrounding images
		self.bg_filter_names = [f[0] for f in bg_filters]

		# all filters
		self.filters = odict(bg_filters + [
			('Original', imdb.pipeline()),
			('Greyscale', imdb.pipeline().grey()),
			('Edges', imdb.pipeline().grey().pipe(lambda im: cv2.Laplacian(im,cv2.CV_64F)).invert()),

			# https://www.learnopencv.com/non-photorealistic-rendering-using-opencv-python-c/
			('Stylization', imdb.pipeline().pipe(lambda im: cv2.stylization(im, sigma_s=10, sigma_r=0.4))),
			('Pencil Sketch', imdb.pipeline().pipe(lambda im: cv2.pencilSketch(im, sigma_s=10, sigma_r=0.1, shade_factor=0.02)[1])),
			('Detail Enhance', imdb.pipeline().pipe(lambda im: cv2.detailEnhance(im, sigma_s=20, sigma_r=0.15))),
			('Edge Preserving', imdb.pipeline().pipe(lambda im: cv2.edgePreservingFilter(im, flags=1, sigma_s=30, sigma_r=0.4))),
		])


	def process_image(self, filename, filter=None):
		filter = filter or 'Original'
		if filter not in self.filters:
			return None

		# get surrounding images for background subtraction filters
		window = self.imdb.cfg.BG.WINDOW if filter in self.bg_filter_names else None

		# get image
		img = self.filters[filter].feed(src=filename, window=window).first()

		# convert from bgr2rgb
		if len(img.shape) > 2:
			img = img[:,:,::-1]

		# fake cropping. blacking out outside the crop. this lets us keep the original aspect ratio/label positioning
		y, x = img.shape[:2]
		if self.imdb.cfg.CROP.Y1:
			img[:int(y*self.imdb.cfg.CROP.Y1),:] = 0
		if self.imdb.cfg.CROP.Y2:
			img[int(y*self.imdb.cfg.CROP.Y2):,:] = 0

		return img

