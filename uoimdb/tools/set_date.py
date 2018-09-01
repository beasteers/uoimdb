import os
import time
import argparse
from shutil import copyfile
from datetime import timedelta, datetime


def set_date(file, date):
	date = datetime.strptime(date, '%m/%d/%Y %H:%M:%S')
	timestamp = time.mktime(date.timetuple())
	os.utime(file, (timestamp, timestamp))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='modify a files timestamp.')
	parser.add_argument('file', help='the image file to use')
	parser.add_argument('-d', '--date', help='the desired date', default=None)
	args = parser.parse_args()

	set_date(file=args.file, date=args.date)


