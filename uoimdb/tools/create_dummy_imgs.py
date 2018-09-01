import os
import time
import glob
import random
import argparse
from shutil import copyfile
from datetime import datetime
from utils import parse_freq



def add_file_suffix(fname, suffix):
	return '{1}{0}{2}'.format(suffix, *os.path.splitext(fname))





def create_imgs(file, start=None, end=None, freq='1h', n=100, output=None):
	print(file, start, end, freq, n, output)
	files = glob.glob(file) if not os.path.isfile(file) else [file]
	if not len(files):
		return

	# split up directory and filename
	files = [
		(os.path.dirname(f), os.path.basename(f), f)
		for f in files
	]

	# convert string to timedelta
	freq = parse_freq(freq)
	if not freq:
		raise ValueError('Invalid frequency argument.')

	# convert start/end times to datetime
	if start and not isinstance(start, datetime):
		start = datetime.strptime(start, '%d/%m/%Y')

	if end and not isinstance(end, datetime):
		end = datetime.strptime(end, '%d/%m/%Y')


	# default values for start and end
	if not start:
		if end and n:
			start = end - n*freq
		else:
			start = datetime.now()

	if not end:
		if start and n:
			end = start + n*freq


	time_range = end - start
	if time_range:
		n = time_range.total_seconds() / freq.total_seconds()

	print(files, start, end, freq, n)
	for i in range(int(n)):
		directory, fname, file = random.choice(files)

		# copy the file
		file_i = os.path.join(output or directory, add_file_suffix(fname, '_{}'.format(i)))
		copyfile(file, file_i)

		# modify the timestamp
		timestamp = start + i*freq
		timestamp = time.mktime(timestamp.timetuple())
		os.utime(file_i, (timestamp, timestamp))








if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create a set of identical images with incremental timestamps.')
	parser.add_argument('file', help='the image file to use')
	parser.add_argument('-s', '--start', help='the start date', default=None)
	parser.add_argument('-e', '--end', help='the end date', default=None)
	parser.add_argument('-n', type=int, help='number of images to produce', default=10)
	parser.add_argument('-f', '--frequency', help='the frequency to increment the timestamps at', default='1h')
	parser.add_argument('-o', '--output', help='output directory', default=None)
	args = parser.parse_args()

	create_imgs(file=args.file, start=args.start, end=args.end, freq=args.frequency, n=args.n, output=args.output)


