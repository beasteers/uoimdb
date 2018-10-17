from __future__ import print_function
import sys
sys.path.insert(1, '..')

import os
import cv2
import json
import glob
import time
import numpy as np
import pandas as pd
from datetime import datetime

from flask import Flask
from flask import request, render_template, jsonify, make_response, send_file, url_for, redirect, flash, abort
import flask_login
import jinja2

import uoimdb as uo
from .image_processing import ImageProcessor
from uoimdb.config import get_config
from uoimdb import utils



def user_col(col, user=None):
    '''Gets the name of a user specific column'''
    if user is None:
        user = flask_login.current_user.get_id() if flask_login.current_user else ''
    return '{}|{}'.format(user, col)

def remove_user_col(col):
    '''Gets the name of a user specific column'''
    return col.split('|')[-1]


from uoimdb.tools.random_sample import RandomSamples


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

        # # store number of times an image was viewed
        # if not 'views' in self.imdb.df.columns:
        # 	self.imdb.df['views'] = 0


        print('Image Database Shape: {}'.format(self.imdb.df.shape))
        print(self.imdb.df.head())

        

        # load all labels
        utils.ensure_dir(self.cfg.LABELS_LOCATION)

        self.labels_dfs = {}
        for f in glob.glob(os.path.join(self.cfg.LABELS_LOCATION, '*.csv')):
            lbl_name = os.path.splitext(os.path.basename(f))[0]
            self.labels_dfs[lbl_name] = pd.read_csv(f).set_index('id').fillna('')

        if not len(self.labels_dfs):
            self.new_label_set_df(self.cfg.DEFAULT_LABELS_FILE)


        # performs the image processing
        self.image_processor = image_processor_class(self.imdb)
        utils.ensure_dir(self.cfg.IMAGE_CACHE_LOCATION)

        # used for finding consecutive days
        self.dates_list = list(np.sort(np.unique(self.imdb.df.date.dt.strftime(cfg.DATE_FORMAT))))


        # expose the configuration options to javascript so it can properly construct links
        self.app.context_processor(self.inject_globals)

        self.define_routes()


        # load all generated random samples
        self.random_samples = RandomSamples(self.imdb, self.image_processor)
        self.random_samples.load_samples() 


    def inject_globals(self):
        return dict(config=self.cfg.frontend,
            current_user=flask_login.current_user.get_id(), 
            image_filters=list(self.image_processor.filters.keys()),
            random_samples=list(self.random_samples.samples.keys()), 
            label_sets=list(self.labels_dfs.keys()),)



    @property
    def current_label_set(self):
        return request.values.get('label_set') or request.cookies.get('label_set') or self.cfg.DEFAULT_LABELS_FILE
    	


    def define_routes(self):
        # define all of the routes
        self.authentication_routes()
        self.page_routes()
        self.random_sample_routes()
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
            if not user.is_authenticated:
                return
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
        @flask_login.login_required
        def index():
            '''Page that lists all images grouped by month/day'''
            return render_template('index.j2', title=self.cfg.APP_TITLE, calendar=True)

        @self.app.route('/labels/')
        @self.app.route('/labels/<name>/')
        @flask_login.login_required
        def label_list(name=None):
            '''Page that lists all images grouped by month/day'''
            name = name or request.cookies.get('label_set') or self.cfg.DEFAULT_LABELS_FILE

            if name not in self.labels_dfs:
                return abort(404, 'This label set does not exist. Use /labels/{}/create/ to create that label set.'.format(name))

            df = self.labels_dfs[name].reset_index()

            stats = df.groupby('label').count().id.rename(lambda l: 'Labeled as {}'.format(l))
            stats = stats.append(
                    df.groupby('user').count().id.rename(lambda u: 'Labeled by {}'.format(u)))

            return render_template('table.j2', title='Annotations in {}'.format(name), 
                columns=df.columns, data=df.to_dict(orient='records'), stats=stats.to_dict(),
                links=[
                    {'name': name, 'url': url_for('label_list', name=name)}
                    for name in self.labels_dfs
                ], 
                set_cookie_button=(dict(name='label_set', value=name, label='Set as Default') 
                                    if name != request.cookies.get('sample_name') else None)
                )


        @self.app.route('/labels/<name>/create/')
        @flask_login.login_required
        def create_label_set(name=None):
            '''Page that lists all images grouped by month/day'''
            if name in self.labels_dfs and request.args.get('overwrite') is None:
                return abort(500, 'Label set {} already exists. Add ?overwrite to overwrite the label set.'.format(name))

            self.new_label_set_df(name)
            return redirect(url_for('label_list', name=name))



        @self.app.route('/video/<date>/')
        @self.app.route('/video/<date>/<time_range>')
        @flask_login.login_required
        def video_day(date, time_range=None):
            '''Display images on a certain day'''
            time_range = time_range or '-'.join(map(str, self.cfg.FILTER_TIME_RANGE))

            return render_template('video.j2', title='{}'.format(date), 
                query=url_for('get_images', date=date, time_range=time_range))


        @self.app.route('/has-labels')
        @self.app.route('/has-labels/<date>')
        @self.app.route('/has-labels/<date>/<time_range>')
        @flask_login.login_required
        def video_all_labels(date=None, time_range=None):
            '''Display images on a certain day'''

            return render_template('video.j2', title='Only Images With Labels', date=date,
                query=url_for('get_images', date=date, time_range=time_range, has_labels=1)) # 


    def new_label_set_df(self, name):
        '''Create a new set of labels'''
        self.labels_dfs[name] = df = pd.DataFrame([], columns=['id', 'src', 'x', 'y', 'w', 'h']).set_index('id').fillna('')
        df.to_csv(os.path.join(self.cfg.LABELS_LOCATION, '{}.csv'.format(name)))



    def random_sample_routes(self):

        @self.app.route('/random/all/')
        @flask_login.login_required
        def random_video():
            '''Display images on a certain day'''
            return render_template('video.j2', title='Random', 
                query=url_for('get_images', random=1))


        @self.app.route('/random/')
        @self.app.route('/random/<name>/')
        @self.app.route('/random/<name>/p/<int:page>')
        @self.app.route('/random/<name>/user/<user>/p/<int:page>')
        @flask_login.login_required
        def random_sample_list(name=None, page=1, user=None):
            '''Display a list of all images in a specific random sample.
            Request Arguments:
                name (str): the sample name.
                page (int, >=1): the list page number
                has_status (str): all images with a certain status for the current user
                <user>_has_status (str): all images with a certain status for a specific user
                not_status (str): all images without a certain status for the current user
                <user>_not_status (str): all images without a certain status for a specific user
                only_cached (bool): if true, only return images that have been cached
                chronological (bool): order images by chronological order
                per_page (int): override number of items per page
                unordered (bool): don't apply any user specific ordering
                
            '''
            if not name:
                try:
                    name = next(iter(self.random_samples.samples.keys()))
                    return redirect(url_for('random_sample_list', name=name))
                except StopIteration:
                    return abort(404, 'Create a random sample by going to /random/<your_new_sample_name>')

            if name not in self.random_samples.samples:
                return abort(404, "{} doesnt exist. Create a random sample by going to /random/{}/create".format(name, name))



            # select random sample from imdb and shuffle order unique for a user.
            if request.args.get('unordered') is None:
                df = self.random_samples.user_sample_order(name, user)
            else:
                df = self.random_samples.samples[name]

            sample_cols = df.columns
            total_count = len(df)
            
            # add src as column and numerical index column
            df = df.copy().reset_index().reset_index()
            
            # filter by status
            has_status = request.args.get('has_status')
            if has_status is not None:
                df = df[df[user_col('status')] == has_status]
                        
            not_status = request.args.get('not_status')
            if not_status is not None:
                df = df[df[user_col('status')] != has_status]
                        
            for username in self.cfg.USERS:
                user_has_status = request.args.get('{}_has_status'.format(username))
                if user_has_status is not None:
                    df = df[df[user_col('status', username)] == user_has_status]
                            
                user_not_status = request.args.get('{}_not_status'.format(username))
                if user_not_status is not None:
                    df = df[df[user_col('status', username)] != user_not_status]
            
            # order by date
            chronological = request.args.get('chronological')
            if chronological is not None:
                df = df.set_index('src').loc[self.imdb.df.date[df.src].sort_values().index].reset_index()
                        
            # filter out any images that have not yet been cached
            if request.args.get('only_cached') is not None:
                df['is_cached'] = df.src.apply(lambda src: 
                    os.path.isfile(self.image_processor.cache_filename(src, self.cfg.DEFAULT_FILTER, ext=self.cfg.IMAGE_CACHE_EXT)))
                df = df[df.is_cached]

            # select images on the specified page
            max_page, per_page = None, int(request.args.get('per_page', self.cfg.TABLE_PAGINATION or 0))
            if per_page:
                max_page = -(-len(df) // per_page) # rounds up using negative
                if 1 <= page <= max_page:
                    df = df.iloc[(page - 1) * per_page : page * per_page]
                else:
                    return abort(404, "Page {} doesn't exist for sample {} with {} images, showing {} per page. "
                                       "Specify a page number between {} and {}.".format(
                                            page, name, total_count, per_page, 1, max_page))

            # mark images in page as cached if not done already.
            if 'is_cached' not in df.columns: 
                # avoid computing this for all images every time. only the current page. unless we need to filter by it.
                df['is_cached'] = df.src.apply(lambda src: 
                    os.path.isfile(self.image_processor.cache_filename(src, self.cfg.DEFAULT_FILTER, ext=self.cfg.IMAGE_CACHE_EXT)))


            # add image dates
            df['date'] = self.imdb.df.date[df.src].values

            # link to image page
            df.src = [
                ('<a href="{}">{}</a>'.format(url_for('random_sample_video', name=name, i=i), row.src)
                 + (' <i class="badge badge-dark" title="Image Is Cached"><i class="fas fa-check"></i></i>' 
                    if row.is_cached else '')
                ) for i, row in df.iterrows()
            ]

            # get rid of columns we don't want to show.
            df = df.drop('is_cached', 1) 

            # order the columns correctly
            main_cols = ['index', 'src', 'date']
            df = df[main_cols + 
                    [c for c in sample_cols if c.startswith(user_col(''))] + 
                    [c for c in sample_cols if not c.startswith(user_col(''))]]


            stats = pd.Series([])
            stats['Total in Sample'] = total_count
            for u in self.cfg.USERS:
                stats[user_col('reviewed', u)] = (df[user_col('status', u)] == 'reviewed').sum()



            page_data = dict(
                title='Random Sample: {}'.format(name) + ('|Page {}/{}'.format(page, max_page) if per_page else ''), 
                name=name, columns=df.columns, stats=stats.to_dict(), 
                data=df.to_dict(orient='records'), 
                prev_query=url_for('random_sample_list', name=name, page=page-1, user=user) if per_page and page > 1 else None,
                next_query=url_for('random_sample_list', name=name, page=page+1, user=user) if per_page and page < max_page else None,
            )


            return render_template('table.j2', 
                links=[
                    {'name': name, 'url': url_for('random_sample_list', name=name)}
                    for name in self.random_samples.samples
                ], 
                set_cookie_button=(dict(name='sample_name', value=name, label='Set as Default') 
                                    if name != request.cookies.get('sample_name') else None),
                **page_data)


        @self.app.route('/random/create/')
        @self.app.route('/random/<name>/create/')
        @self.app.route('/random/<name>/create/<int:n>/')
        @flask_login.login_required
        def create_random_sample(name, n=self.cfg.SAMPLE_SIZE):
            '''Display images on a certain day
            Request Parameters:
                name: the sample name
                n (int): the number in a sample

                overlap_existing (flag, 1): use to allow samples to overlap
                n_samples (int): create n samples that overlap by a certain percentage
                overlap_ratio (float): how much of the n_samples should overlap

            '''
            if not name:
                return abort(500, 'You must specify a sample name')

            time_range = request.args.get('time_range')
            if time_range:
                time_range = map(int, time_range.split('-'))

            distance_to_gap = request.args.get('distance_to_gap')
            if distance_to_gap is not None:
                distance_to_gap = int(distance_to_gap)

            overlap_ratio = request.args.get('overlap_ratio')
            if overlap_ratio is not None:
                overlap_ratio = float(overlap_ratio)

            names = self.random_samples.create_sample(name, n=n, 
                overlap_existing=request.args.get('overlap_existing') is not None,
                n_samples=int(request.args.get('n_samples', 0)),
                overlap_ratio=overlap_ratio,
                time_range=time_range,
                distance_to_gap=distance_to_gap)

            if not names:
                return abort(500, 'Sample {} could not be created.'.format(name))

            name = names[0]
            return redirect(url_for('random_sample_list', name=name))



        @self.app.route('/random/<name>/delete/')
        @flask_login.login_required
        def delete_random_sample(name):
            random_sample.delete_sample(name)
            return redirect(url_for('random_sample_list'))


        @self.app.route('/random/<name>/i/<int:i>/')
        @self.app.route('/random/<name>/user/<user>/i/<int:i>/')
        @flask_login.login_required
        def random_sample_video(name, i, user=None):
            '''Display images on a certain day'''
            if name in self.random_samples.samples and i < len(self.random_samples.samples[name]):
                # src = self.random_samples.samples[name].index[i]
                # if not self.random_samples.samples[name].loc[src, user_col('status')]:
                # 	self.random_samples.samples[name].loc[src, user_col('status')] = 'reviewed'

                return render_template('video.j2', title='Random', sample_name=name,
                    query=url_for('get_images', sample_name=name, sample_index=i, sample_user=user),
                    prev_query=url_for('random_sample_video', name=name, i=i-1, sample_user=user) if i-1 >= 0 else None,
                    next_query=url_for('random_sample_video', name=name, i=i+1, sample_user=user) if i+1 < len(self.random_samples.samples[name]) else None,
                    parent_page=url_for('random_sample_list', name=name))
            else:
                return abort(404)
        

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
            lbl_name = self.current_label_set
            labels = self.labels_dfs[lbl_name]
            nlabels_prev = len(labels)

            boxes = request.form.get('boxes')
            if boxes:
                self.add_labels(json.loads(boxes))
                self.labels_dfs[lbl_name].to_csv(os.path.join(self.cfg.LABELS_LOCATION, '{}.csv'.format(lbl_name)))

            # save if image has been seen
            name = request.form.get('sample_name')
            img_meta = request.form.get('img_meta')
            if name and img_meta:
                img_meta = json.loads(img_meta)
                for src, meta in img_meta.items():
                    if 'status' in meta:
                        self.random_samples.samples[name].loc[src, user_col('status')] = meta['status']

                self.random_samples.save_user_sample(name)

            return jsonify({'message': 'Saved! &#128077;', 'nlabels': len(labels), 'nlabels_prev': nlabels_prev})


        @self.app.route('/clear-images')
        @flask_login.login_required
        def clear_image_data():
            self.imdb.clear_images() # clears all loaded image data
            return redirect(url_for('calendar'))


        @self.app.route('/query_images')
        @flask_login.login_required
        def get_images():
            '''retrieve images using some constraints'''

            # parse request
            src = request.args.get('src')
            window = request.args.get('window')
            window = map(int, window.split(',')) if window else self.cfg.SAMPLE_WINDOW
            lwindow, rwindow = window

            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            date = request.args.get('date')
            time_range = request.args.get('time_range')

            random = request.args.get('random')
            has_labels = request.args.get('has_labels')
            viewed = request.args.get('viewed')

            sample_name = request.args.get('sample_name')
            sample_index = request.args.get('sample_index')
            # sample_user = request.args.get('sample_user')
            print(request.args)

            # start with all images and filter based on request
            df = self.imdb.df.drop('im', 1)

            i_center = None
            if sample_name:
                sample = self.random_samples.user_sample_order(sample_name, user=request.args.get('sample_user'))

                if sample_index is None:
                    sample_index = np.where(sample[user_col('status')] != 'reviewed')[0]
                sample_index = int(sample_index)
                src = sample.index[sample_index]

            if src:
                i = df.index.get_loc(src)
                df = df.iloc[i - lwindow:i + 1 + rwindow]
                i_center = lwindow

            else:
                if date:
                    df = df[df.date.dt.date == pd.to_datetime(date).date()]
                else:
                    if start_date:
                        df = df[df.date.dt.date >= pd.to_datetime(start_date).date()]
                    if end_date:
                        df = df[df.date.dt.date < pd.to_datetime(end_date).date()]

                if time_range:
                    morning, evening = time_range.split('-')
                    if morning:
                        df = df[df.date.dt.hour >= int(morning)]
                    if evening:
                        df = df[df.date.dt.hour < int(evening)]	

                labels = self.labels_dfs[self.current_label_set]
                if has_labels == '1':
                    df = df[df.index.isin(labels.src)]
                elif has_labels == '0':
                    df = df[~df.index.isin(labels.src)]

                if random:
                    i = np.random.randint(lwindow, len(df) - rwindow) # get random row constrained by window.
                    df = df.iloc[i - lwindow:i + 1 + rwindow]
                    i_center = lwindow

            if sample_name:
                common_idx = df.index.intersection(sample.index)
                for c in sample.columns:
                    if c.startswith(user_col('')):
                        df.loc[common_idx, remove_user_col(c)] = sample.loc[common_idx, c]


            # for all filtered images, get bounding boxes
            timeline = self.get_boxes_for_imgs(df)
            timeline.date = timeline.date.dt.strftime(self.cfg.DATETIME_FORMAT) # make json serializable

            # build queries for prev/next buttons
            prev_query, next_query = {}, {}

            if sample_name:
                if sample_index - 1 >= 0:
                    prev_query['sample_name'] = sample_name
                    prev_query['sample_index'] = sample_index - 1
                if sample_index + 1 < len(df):
                    next_query['sample_name'] = sample_name
                    next_query['sample_index'] = sample_index + 1
            elif src:
                if i_center - 1 >= 0:
                    prev_query['src'] = df.index[i_center-1]
                if i_center + 1 < len(df):
                    next_query['src'] = df.index[i_center+1]

            if date:
                prev_query['date'], next_query['date'] = self.get_next_prev_date(date)

            if random:
                next_query['random'] = 1
                if prev_query:
                    prev_query['random'] = 1


            return jsonify(dict(
                timeline=timeline.fillna('').to_dict(orient='records'),
                prev_query=url_for('get_images', **prev_query) if prev_query else None, 
                next_query=url_for('get_images', **next_query) if next_query else None, 
                i_focus=i_center))


        @self.app.route('/calendar-data')
        @flask_login.login_required
        def get_calendar_data():
            labels = self.labels_dfs[self.current_label_set]

            df = self.imdb.df.reset_index().set_index('date')
            df = df.groupby(pd.Grouper(freq='M')
                ).apply(lambda month: month.groupby(month.index.day
                    ).apply(lambda day: pd.Series(dict(
                            image_count=len(day),
                            label_count=sum(labels.src.isin(day.src)),
                            #view_count=sum(day.views > 0),
                            date=day.index[0].strftime(self.cfg.DATE_FORMAT)
                        )).to_dict()
                    ).to_dict()
                )
            df = df[df.apply(len) > 0]
            df.index = df.index.strftime('%Y/%m')
            return jsonify(df.to_dict())


    def image_routes(self):
        '''

        Image filters
        Defined in image_processing.py

        '''

        @self.app.route('/filter/<filter>/<path:filename>')
        @flask_login.login_required
        def filtered_image(filter, filename):
            # t = time.time()
            cache_filename = self.image_processor.cache_filename(filename, filter)

            if os.path.isfile(cache_filename):
                img = cv2.imread(cache_filename, -1)
                assert img is not None
                # print('loaded cache', time.time() - t)
            else:
                img = self.image_processor.process_image(filename, filter)
                assert img is not None
                # print('processed', time.time() - t)
                if request.args.get('cache_result'):
                    cv2.imwrite(cache_filename, img)

            img = self.image_processor.pre_render_processing(img)
            # print(img.shape, 'total time', time.time() - t)
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
        labels = self.labels_dfs[self.current_label_set]
        labels = labels[labels.src.isin(df.index)]

        if len(labels):
            df['boxes'] = labels.reset_index().groupby('src').apply(lambda s: s.to_dict(orient='records')).rename('boxes')
            df = df.reset_index().rename(columns={'index': 'src'})
            df.boxes = df.boxes.fillna('')#.apply(lambda x: () if pd.isna(x) else x)
        else:
            df = df.reset_index()
            df['boxes'] = ''
        return df


    def get_timeline(self, date, freq=None, imgs_per_group=100):
        day_df = self.imdb.df[self.imdb.df.date.dt.strftime(self.cfg.DATE_FORMAT) == date].reset_index()
        
        if freq is None:
            freq = '{}s'.format(int(min(1. * imgs_per_group / len(day_df), 1) * 24*60*60)) #'6h'

        labels = self.labels_dfs[self.current_label_set]

        timeline = day_df.groupby(pd.Grouper(key='date', freq=freq)
        ).apply(lambda imgs: pd.Series(dict(
            image_count=len(imgs),
            #boxes=imgs.merge(labels_df, on='src').to_dict(orient='records'),
            label_count=sum(labels.src.isin(imgs.src)),
            srcs=imgs['src'].tolist()
        ))).reset_index()

        timeline.date = timeline.date.dt.strftime('%H:%M')
        #timeline.boxes = timeline.boxes.fillna('')
        #timeline['label_count'] = timeline.boxes.apply(lambda boxes: len(boxes))
        return timeline


    def get_next_prev_date(self, date):
        '''Gets the previous/next dates that are available with images'''
        i = self.dates_list.index(date)
        prev_date = self.dates_list[i - 1] if i > 0 else None,
        next_date = self.dates_list[i + 1] if i >= 0 and i < len(self.dates_list)-1 else None
        return prev_date, next_date


    def add_labels(self, new_labels, name=None):
        '''add new records of labels to table.
        Arguments:
            new_labels (list of dicts): each dict containing labels_df columns representing one bounding box
        '''
        if not len(new_labels):
            return

        name = name or self.current_label_set

        df = pd.DataFrame.from_dict(new_labels).set_index('id')
        df['user'] = flask_login.current_user.get_id()

        df = df.combine_first(self.labels_dfs[name]).fillna('') # merge
        self.labels_dfs[name] = df[df['src'] != False] # remove deleted boxes


    def run(self, **kw):
        kw = kw or self.cfg.APP_RUN
        self.app.run(**kw)


def image_response(output_img, ext='.jpg'):
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







	
