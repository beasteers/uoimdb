from __future__ import print_function
import sys
sys.path.insert(1, '..')

import os
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime

from flask import Flask, Blueprint
from flask import request, render_template, jsonify, make_response, url_for, redirect, flash
import flask_login
import jinja2

import uoimdb as uo
from .image_processing import ImageProcessor
from uoimdb.config import get_config


APP_ROOT = os.path.dirname(os.path.realpath(__file__))

class TaggingApp(object):

	def __init__(self, cfg=None, name=__name__, image_processor_class=ImageProcessor):
		'''

		Launch App

		'''

		self.cfg = cfg = get_config(cfg)
		self.app = Flask(name or self.__class__.__name__, template_folder=os.path.join(APP_ROOT, 'templates'))
		self.app.secret_key = cfg.SECRET_KEY
		if cfg.TEMPLATE_DIRECTORIES:
			app.jinja_loader = jinja2.ChoiceLoader([app.jinja_loader] + [
				jinja2.FileSystemLoader(path) 
				for path in cfg.TEMPLATE_DIRECTORIES
			])
		self.app.wsgi_app = PrefixMiddleware(self.app.wsgi_app, prefix=self.cfg.BASE_URL)


		self.login_manager = flask_login.LoginManager()
		self.login_manager.init_app(self.app)


		'''

		Data Storage
		- loading image paths and dates - used to store currently loaded images
		- loading and storing labels - currently via csv, but that can change
		- loading image processor used for all image pipelines

		'''

		self.imdb = uo.uoimdb(cfg=cfg)
		if not len(self.imdb.df):
			raise SystemExit("No images found at {}... \(TnT)/".format(self.imdb.abs_file_pattern))

		print('Image Database Shape: {}'.format(self.imdb.df.shape))
		print(self.imdb.df.head())


		# dataframe to store new labels in and output csv file for labels
		if os.path.isfile(cfg.LABELS_FILE):
			self.labels_df = pd.read_csv(cfg.LABELS_FILE).set_index('id').fillna('')
		else:
			self.labels_df = pd.DataFrame([], columns=['id', 'src', 'x', 'y', 'w', 'h']).set_index('id').fillna('')

		# performs the image processing
		self.image_processor = image_processor_class(self.imdb)

		# used for finding consecutive days
		self.dates_list = list(np.sort(np.unique(self.imdb.df.date.dt.strftime(cfg.DATE_FORMAT))))


		# expose the configuration options to javascript so it can properly construct links
		self.app.context_processor(self.inject_globals)

		self.define_routes()


	def inject_globals(self):
		return dict(config=self.cfg.frontend,
			current_user=flask_login.current_user.get_id(), 
			filters=list(self.image_processor.filters.keys()))	


	def define_routes(self):
		# define all of the routes
		self.authentication_routes()
		self.page_routes()
		self.ajax_routes()
		self.image_routes()



	def authentication_routes(self):
		'''

		User Authentication

		'''
		# login setup
		class User(flask_login.UserMixin):
			is_authenticated = True

		@self.login_manager.user_loader
		def user_loader(id):
			if id not in self.cfg.USERS:
				return

			user = User()
			user.id = id
			return user

		@self.login_manager.request_loader
		def request_loader(request):
			id = request.form.get('userid')
			user = user_loader(id)
			if not user:
				return
			
			pwd = request.form.get('password')
			if not pwd:
				return

			user.is_authenticated = pwd == self.cfg.USERS[id]
			return user

		@self.app.route('/login', methods=['GET', 'POST'])
		def login():
			if request.method == 'POST':
				user = request_loader(request)
				if user:
					flask_login.login_user(user)
					return redirect(url_for(request.args.get('next', 'calendar')))
				flash('permission denied.')
			return render_template('login.j2')


		@self.app.route('/logout')
		def logout():
			flask_login.logout_user()
			return redirect(url_for('login'))


		@self.login_manager.unauthorized_handler
		def unauthorized_handler():
			return redirect(url_for('login'))



	def page_routes(self):
		'''

		Pages
		 - calendar (/, /cal): renders the calendar page
		 - timeline (/cal/<date>): batch image labeling for one day
		 - label_list (/label-list): shows the stored labels in a table

		'''

		@self.app.route('/')
		@self.app.route('/cal/')
		@flask_login.login_required
		def calendar():
			'''Page that lists all images grouped by month/day'''
			return render_template('calendar.j2', title='Calendar', calendar=self.get_calendar())


		@self.app.route('/cal/<date>/')
		@flask_login.login_required
		def timeline_day(date):
			'''Display images on a certain day'''
			timeline = self.get_timeline(date)
			prev_day, next_day = self.get_next_prev_date(date)

			return render_template('tagger.j2', title='{}'.format(date), 
				timeline=timeline.to_dict(orient='records'),
				prev_day=prev_day, next_day=next_day, date=date) # 


		@self.app.route('/video/<date>/')
		@self.app.route('/video/<date>/<time_range>')
		@flask_login.login_required
		def video_day(date, time_range='5-18'):
			'''Display images on a certain day'''
			
			# get dataframe of [src, date, label_count]
			im_dates = self.imdb.df.date[self.imdb.df.date.dt.strftime(self.cfg.DATE_FORMAT) == date]

			morning, evening = time_range.split('-')
			if morning:
				im_dates = im_dates[im_dates.dt.hour >= int(morning)]
			if evening:
				im_dates = im_dates[im_dates.dt.hour < int(evening)]		

			timeline = self.get_boxes_for_imgs(im_dates.dt.strftime(self.cfg.DATETIME_FORMAT))
			prev_day, next_day = self.get_next_prev_date(date)

			return render_template('video.j2', title='{}'.format(date), 
				timeline=timeline.to_dict(orient='records'),
				prev_day=prev_day, next_day=next_day, date=date)


		@self.app.route('/has-labels')
		@self.app.route('/has-labels/<date>')
		@self.app.route('/has-labels/<date>/<time_range>')
		@flask_login.login_required
		def video_all_labels(date=None, time_range=None):
			'''Display images on a certain day'''

			# get dataframe of [src, date, label_count]
			df = self.imdb.df.date[self.imdb.df.index.isin(self.labels_df.src)]

			if date:
				df = df[self.imdb.df.date.dt.strftime(self.cfg.DATE_FORMAT) == date]

			morning, evening = time_range.split('-') if time_range else (None, None)
			if morning:
				df = df[df.dt.hour >= int(morning)]
			if evening:
				df = df[df.dt.hour < int(evening)]

			timeline = self.get_boxes_for_imgs(df.dt.strftime(self.cfg.DATETIME_FORMAT))
			prev_day, next_day = self.get_next_prev_date(date) if date else (None, None)

			return render_template('video.j2', title='Only Images With Labels',
				timeline=timeline.to_dict(orient='records'),
				prev_day=prev_day, next_day=next_day, date=date) # 



		@self.app.route('/label-list/')
		@flask_login.login_required
		def label_list():
			'''Page that lists all images grouped by month/day'''
			df = self.labels_df.reset_index()
			return render_template('table.j2', title='Collected Annotations', 
				columns=df.columns, 
				data=df.to_dict(orient='records'))


	def ajax_routes(self):
		'''

		Ajax Page actions:
		 - save (/save): save new labels - should be list with same columns as labels_df
		 - select_images (/select-images): preload images and get the existing labels
		 - filtered_image (/filter/<filter>/<filename>): get image with a specific applied (i.e. /filter/Edges/unique/path/to/image.png)
		
		'''

		@self.app.route('/save/', methods=['POST'])
		@flask_login.login_required
		def save():
			'''Post the bounding boxes to save'''
			# requests don't like nested request bodies for some reason, so data is double encoded
			nlabels_prev = len(self.labels_df)
			self.add_labels(json.loads(request.form['boxes']))
			self.labels_df.to_csv(self.cfg.LABELS_FILE)

			return jsonify({'message': 'Saved!', 'nlabels': len(self.labels_df), 'nlabels_prev': nlabels_prev})


		@self.app.route('/select-images/')
		@flask_login.login_required
		def select_images():
			'''select a new group of images. get bounding boxes'''
			srcs = json.loads(request.args.get('srcs', "[]"))

			df = self.imdb.df[self.imdb.df.index.isin(srcs)].drop('im', 1).reset_index()
			df['boxes'] = df.index.map(lambda src: 
				(self.labels_df[self.labels_df['src'] == src]
							.reset_index()
							.fillna('')
							.to_dict(orient='records')))

			return jsonify(df.to_dict(orient='records') if len(df) else [])


		@self.app.route('/clear-images')
		@flask_login.login_required
		def clear_image_data():
			self.imdb.clear_images() # clears all loaded image data
			return redirect(url_for('calendar'))



	def image_routes(self):
		'''

		Image filters
		Defined in image_processing.py

		'''

		@self.app.route('/filter/<filter>/<path:filename>')
		@flask_login.login_required
		def filtered_image(filter, filename):
			img = self.image_processor.process_image(filename, filter)
			if img is None:
				img = np.array([]) # empty image

			return image_response(img)

		@self.app.route('/random-image')
		@self.app.route('/random-image/<filter>')
		@flask_login.login_required
		def random_image(filter='Original'):
			src = self.imdb.df.sample(1).index[0]
			img = self.image_processor.filters[filter].feed(src=src).first()
			return image_response(img)


	'''

	Utility functions
	- get_calendar: return images grouped by month/year then day
	- get_timeline: return images grouped into batches based on time - 
					either specify a frequency (i.e. 6h, 30m) or an approx count per batch
	- get_next_prev_date: return the date before and after the specified date - used for next/prev buttons
	- add_labels: update labels table based on incoming labels
	- image_response: convert opencv image to flask response

	'''


	def get_boxes_for_imgs(self, df):
		labels = self.labels_df[self.labels_df.src.isin(df.index)]
		if len(labels):
			boxes = labels.reset_index().groupby('src').apply(lambda s: s.to_dict(orient='records')).rename('boxes')
			timeline = pd.concat([df, boxes], axis=1, sort=False).reset_index().rename(columns={'index': 'src'})
			timeline.boxes = timeline.boxes.fillna('')
		else:
			timeline = df.reset_index()
			timeline['boxes'] = ''
		return timeline


	def get_calendar(self):
		'''gets image stats first by month/year then by day'''
		df = self.imdb.df.reset_index().set_index('date')
		df = df.groupby(pd.Grouper(freq='M')
		).apply(lambda month: month.groupby(month.index.day
			).apply(lambda day: pd.Series(dict(
					image_count=len(day),
					label_count=sum(self.labels_df.src.isin(day.src)),
					date=day.index[0].strftime(self.cfg.DATE_FORMAT)
				)).to_dict()
			).to_dict()
		)
		df = df[df.apply(len) > 0]
		df.index = df.index.strftime('%Y/%m')
		return df.to_dict()


	def get_timeline(self, date, freq=None, imgs_per_group=100):
		day_df = self.imdb.df[self.imdb.df.date.dt.strftime(self.cfg.DATE_FORMAT) == date].reset_index()
		
		if freq is None:
			freq = '{}s'.format(int(min(1. * imgs_per_group / len(day_df), 1) * 24*60*60)) #'6h'

		timeline = day_df.groupby(pd.Grouper(key='date', freq=freq)
		).apply(lambda imgs: pd.Series(dict(
			image_count=len(imgs),
			#boxes=imgs.merge(labels_df, on='src').to_dict(orient='records'),
			label_count=sum(self.labels_df.src.isin(imgs.src)),
			srcs=imgs['src'].tolist()
		))).reset_index()

		timeline.date = timeline.date.dt.strftime('%H:%M')
		#timeline.boxes = timeline.boxes.fillna('')
		#timeline['label_count'] = timeline.boxes.apply(lambda boxes: len(boxes))
		return timeline

	def get_next_prev_date(self, date):
		'''Gets the previous/next dates that are available with images'''
		i = self.dates_list.index(date)
		return (
			self.dates_list[i - 1] if i > 0 else None,
			self.dates_list[i + 1] if i >= 0 and i < len(self.dates_list)-1 else None
		)


	def add_labels(self, new_labels):
		'''add new records of labels to table.
		Arguments:
			new_labels (list of dicts): each dict containing labels_df columns representing one bounding box
		'''
		if not len(new_labels):
			return

		df = pd.DataFrame.from_dict(new_labels).set_index('id')
		df['user'] = flask_login.current_user.get_id()
		self.labels_df = df.combine_first(self.labels_df).fillna('') # merge
		self.labels_df = self.labels_df[self.labels_df['src'] != False] # remove deleted boxes


	def run(self, **kw):
		kw = kw or self.cfg.APP_RUN
		self.app.run(**kw)





def image_response(output_img, ext='.png'):
	'''Converts opencv image to a flask response.'''
	retval, buffer = cv2.imencode(ext, output_img)
	return make_response(buffer.tobytes())


class PrefixMiddleware(object):
		def __init__(self, app, prefix=''):
				self.app = app
				self.prefix = prefix

		def __call__(self, environ, start_response):
				environ['SCRIPT_NAME'] = self.prefix
				return self.app(environ, start_response)




def run():
	import argparse
	parser = argparse.ArgumentParser(description='UO Annotations App')
	parser.add_argument('--cfg', default=None, # uses default config
						help='the config file')
	args = parser.parse_args()

	# load the specified config file
	cfg = get_config(args.cfg)

	app = TaggingApp(cfg=cfg)
	app.run()



if __name__ == '__main__':
	run()







	
