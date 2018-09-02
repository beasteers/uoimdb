import os
import re
import time
import glob
import random
import argparse
from shutil import copyfile
from datetime import timedelta, datetime
import dateutil.parser as parser

import imageio
from .. import utils
from ..config import get_config

cfg = get_config()



def add_file_suffix(fname, suffix):
	return '{1}{0}{2}'.format(suffix, *os.path.splitext(fname))


def sample_images_from_image(file, start, freq, n, output=None, random_sample=True):
	# split up directory and filename
	files = [
		(os.path.dirname(f), os.path.basename(f), f)
		for f in glob.glob(file)
	]

	for i in range(int(n)):
		if random_sample:
			directory, fname, file = random.choice(files)
		else:
			directory, fname, file = files[i % len(files)]

		# copy the file
		file_i = os.path.join(output or directory, add_file_suffix(fname, '_{}'.format(i)))
		copyfile(file, file_i)

		# modify the timestamp
		timestamp = start + i*freq
		timestamp = time.mktime(timestamp.timetuple())
		os.utime(file_i, (timestamp, timestamp))



def create_images_from_video(file, start, freq, n=None, skip=None, output=None):
	output = output or os.path.dirname(f)
	if not os.path.isdir(output):
		os.makedirs(output)

	file_pattern = os.path.join(output, os.path.splitext(os.path.basename(file))[0] + '_{}' + cfg.SAVE_EXT)

	reader = imageio.get_reader(file)
	for i, im in enumerate(reader):
		if skip:
			if i % skip:
				continue
			else:
				i = i // skip

		if n and i >= n:
			break


		# save image frame
		file_i = file_pattern.format(i)
		imageio.imwrite(file_i, im)

		# modify the timestamp
		timestamp = start + i*freq
		timestamp = time.mktime(timestamp.timetuple())
		os.utime(file_i, (timestamp, timestamp))




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate sample images.')
	parser.add_argument('action', help=(# 'repeat: generate images sequentially by repeating files that match a file pattern; '
										'random: randomly sample from files that match a file pattern; '
										'video: generate images from video frames; '))
	parser.add_argument('file', help='the image file to use')
	parser.add_argument('-s', '--start', help='the start date', default=None)
	parser.add_argument('-e', '--end', help='the end date', default=None)
	parser.add_argument('-n', type=int, help='number of images to produce', default=None)
	parser.add_argument('-f', '--frequency', help='the frequency to increment the timestamps at', default='10m')
	parser.add_argument('--skip', type=int, help='you can sample a video every x frames. ', default=None)
	parser.add_argument('-o', '--output', help='output directory', default=cfg.IMAGE_OUTPUT_DIR)
	args = parser.parse_args()

	start, end, freq, n = args.start, args.end, args.frequency, args.n

	# convert string to timedelta
	freq = utils.parse_freq(freq)
	if not freq:
		raise ValueError('Invalid frequency argument.')

	# convert start/end times to datetime
	if start and not isinstance(start, datetime):
		start = parser.parse(start) #datetime.strptime(start, '%d/%m/%Y')

	if end and not isinstance(end, datetime):
		end = parser.parse(end) #datetime.strptime(end, '%d/%m/%Y')


	# default values for start and end
	if not start:
		if end and n:
			start = end - n*freq
		else:
			start = datetime.now()

	if not end:
		if start and n:
			end = start + n*freq

	if end:
		time_range = end - start
		if time_range:
			n = time_range.total_seconds() / freq.total_seconds()



	if args.action == 'random':
		sample_images_from_image(args.file, start=start, freq=freq, n=n, output=args.output)

	if args.action == 'video':
		create_images_from_video(args.file, start=start, freq=freq, n=n, skip=args.skip, output=args.output)
