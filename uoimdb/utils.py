from __future__ import print_function
from functools import wraps

import re
import os
import sys
import time
import pickle
import numpy as np
from datetime import timedelta

import multiprocessing as mp
import traceback





# Just remembered this can be handled by dataset.py..
# def calc_label_properties(df, corners=False, origin=False, dur=False, refresh=False):
#     '''Calculate some extra label properties that are needed elsewhere'''
#     corners = corners and (not 'x1' in df or
#                            not 'y1' in df or 
#                            not 'x2' in df or 
#                            not 'y2' in df or refresh)
#     dur = dur and (not 'dur' in df or refresh)
#     origin = (origin or dur) and (not 'origin_id' in df or refresh)

#     if corners:
#         df.loc[:,'x1'] = (df.x - df.w/2)
#         df.loc[:,'y1'] = (df.y - df.h/2)
#         df.loc[:,'x2'] = (df.x + df.w/2)
#         df.loc[:,'y2'] = (df.y + df.h/2)

#     if origin or dur:
#         if not 'origin_id' in df or refresh:
#             only_plumes = df.label == 'plume'
#             df.loc[:,'origin_id'] = None
#             df.loc[only_plumes] = df[only_plumes].apply(get_origin_id, axis=1, df=df[only_plumes])

#     if dur and (not 'dur' in df or refresh):
#         plume_dur = df[only_plumes].groupby('origin_id').src.count()
#         df.loc[only_plumes, 'plume_dur'] = plume_dur[df[only_plumes].origin_id].values

#     return df



def null_image(im):
    return im is None or type(im) == float and np.isnan(im) # check the numerous ways images could be bad


'''

Debugging

'''

def progress(iterable, every=1, i=0):
    '''Print out the iteration number of an iterable. For some reason, tqdm can't be installed...'''
    for i, _ in enumerate(iterable, i):
        if i % every == 0:
            print(i, end=' ')
            sys.stdout.flush()
        yield _
    print(i, 'end')
    
    
def timer(iterable, freq=None):
    '''time each iteration of a generator
    Arguments:
        iterable: obvious
        freq: number of iterations before updating on average time/iter
    '''
    times, t = [], time.time()
    for i, _ in enumerate(iterable):
        times.append(time.time() - t)
        if freq and i and i % freq == 0: # report time over the last `freq` iterations
            print('{} +- {}'.format( np.mean(times[-freq:]), np.std(times[-freq:]) ))
        yield _
        t = time.time()
    if len(times): # report the overall time stats
        print('average time:', np.mean(times), 'total time:', np.sum(times),
              'min time:', np.min(times), 'max time:', np.max(times),)



'''

Custom Data Structures

'''

    
class npBuffer(np.ndarray):
    '''Like a deque but for numpy arrays. Stores a rolling mean of the buffer.'''
    def __new__(cls, arr, calc_mean=True):
        arr = np.asarray(arr).view(cls)
        arr.mean_ = np.mean(arr, axis=0) if calc_mean else None
        return arr
        
    def append(self, value):
        if self.mean_ is not None:
            self.mean_ += (value - self[0]) / len(self)
        self[:-1] = self[1:]
        self[-1] = value
        

class metaArray(np.ndarray):
    '''Stores meta data in a numpy array'''
    def set_meta(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self


# class easydict(dict):    
#     '''Dict keys accessible via attributes (for config)'''
#     __delattr__ = dict.__delitem__
#     __setattr__ = dict.__setitem__

#     def __init__(self, d=None):
#         if d is not None:
#             self.update(**d)
            
#     def __getattr__(self, name):
#         self[name] = self.__convert__(self.get(name))
#         return self[name]
        
#     def __convert__(self, value):
#         if not isinstance(value, self.__class__):
#             if isinstance(value, dict):
#                 return self.__class__(value)
#             elif isinstance(value, (list, tuple)):
#                 return [self.__convert__(x) for x in value]
#         return value


class easydict(dict):    
    '''Dict keys accessible via attributes (for config)'''
    __delattr__ = dict.__delitem__
    __getattr__ = dict.get

    def __init__(self, **d):
        self.update(**d)

    def __setattr__(self, name, value):
        return self.__setitem__(name, value)
    
    def __setitem__(self, name, value):
        return dict.__setitem__(self, name, self.__convert__(value))
    
    def update(self, **kw):
        for k, v in kw.items():
            if k in self and isinstance(self[k], dict) and isinstance(v, dict):
                self[k].update(**v)
            else:
                self[k] = v
        return self
        
    def __convert__(self, value):
        if not isinstance(value, self.__class__):
            if isinstance(value, dict):
                return self.__class__(**value)
            elif isinstance(value, (list, tuple)):
                return [self.__convert__(x) for x in value]
        return value
    



'''

Date Utils

'''

def get_path_date(path):
    '''Get the timestamp from a file path'''
    return datetime.fromtimestamp(os.stat(path).st_mtime)
                
def set_date(fname, date):
    '''Set the timestamp of a file'''
    if isinstance(date, str):
        date = datetime.strptime(date, '%m/%d/%Y %H:%M:%S')
    timestamp = time.mktime(date.timetuple())
    os.utime(fname, (timestamp, timestamp)) 


time_units = [('weeks', 'w'), ('days', 'd'), ('hours', 'h'), ('minutes', 'm'), 
              ('seconds', 's'), ('milliseconds', 'ms'), ('microseconds', 'us')]

freq_regex = re.compile(r'\s*'.join([
    r'((?P<{}>\d+?){})?'.format(var, unit) 
    for var, unit in time_units
]))

def parse_freq(freq):
    '''Convert a frequency string to time delta
    Arguments:
        freq (str): of the form <quantity><unit>. Must be ordered from largest unit to smallest
            e.g. 6h is 6 hrs, 6h30m is 6 hrs 30 mins, etc.
    '''
    return timedelta(**{
        name: int(param) for name, param in 
        freq_regex.match(freq).groupdict().items() if param
    })
        
        
        
  

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

        
def cache_result(file, refresh=False, verbose=False):
    '''Cache the result of a class method to a file
    Arguments:
        cache_file (str or callable): value should be path to save/load data to/from.
            Callable will be given all arguments from the target function.
        refresh (bool): whether to refresh by default or not. 
            Can pass _refresh_ to func to override.
        verbose (bool): whether to print saving/loading info. 
            Can pass _verbose_ to func to override.
    
    # class method example
    class A:
        name = 'hello'
        @cache_result(lambda self, *a, **kw: '{}.pkl'.format(self.name))
        def get_stuff(self):
            return 5 # some calculations or whatever
    
    # function example 1
    @cache_result(lambda a, b: '{}+{}.pkl'.format(a, b))
    def a(a, b):
        return a + b # calcs
      
    # function example 2
    @cache_result('b_result.pkl')
    def b():
        return 5 # calcs
    '''
    def outer(func):
        @wraps(func)
        def inner(*a, _refresh_=refresh, _verbose_=verbose, **kw):
            cache_file = file(*a, **kw) if callable(file) else file
            
            # check if cache is available
            if not _refresh_ and os.path.isfile(cache_file):
                with open(cache_file, 'rb') as fid:
                    data = pickle.load(fid)
                if _verbose_:
                    print('{}(..) loaded from {}'.format(func.__qualname__, cache_file))
                return data
            # cache is not available, compute value
            data = func(*a, **kw)
            # save value to cache
            ensure_dir(os.path.dirname(cache_file))
            with open(cache_file, 'wb') as fid:
                pickle.dump(data, fid, pickle.HIGHEST_PROTOCOL)
            if _verbose_:
                print('wrote {}(..) to {}'.format(func.__qualname__, cache_file))
            return data
        return inner
    return outer
        
        
        
        
        
# def dict2xml(data, root='root', key=None):
#     '''Convert dict to xml tree.
#     Arguments:
#         data (dict): the data to convert
#         root (str or xml element): the root element (or element name) to data to.
#         key (for internal use): tag name. used for recursion.
    
#     For multiple entries with the same name, pass as {'tag' ['hi', 'hello']}
#     Renders as:
#         <tag>hi</tag>
#         <tag>hello</tag>
#     '''
#     if isinstance(root, str):
#         root = ET.Element(root)
        
#     if key is not None:
#         root = ET.SubElement(root, key)
    
#     if isinstance(data, dict):
#         for key, val in data.items():
#             if isinstance(val, list):
#                 for v in val:
#                     dict2xml(v, root=root, key=key)
#             else:
#                 dict2xml(val, root=root, key=key)
#     elif data is not None:
#         root.text = str(data)
#     return root



        

    
    
    
    
    
# # https://stackoverflow.com/questions/6728236/exception-thrown-in-multiprocessing-pool-not-detected
# def log_exceptions(func):
#     @wraps(func)
#     def inner(*args, **kwargs):
#         try:
#             result = func(*args, **kwargs)
#         except: # catch everythinggg
#             # send traceback
#             multiprocessing.get_logger().error(traceback.format_exc(), *args)
#             raise Exception # reraise as base exception
#         return result
#     return inner

# class Process(mp.Process):
#     def __init__(self, *args, **kwargs):
#         mp.Process.__init__(self, *args, **kwargs)
#         self._pconn, self._cconn = mp.Pipe()
#         self._exception = None

#     def run(self):
#         try:
#             mp.Process.run(self)
#             self._cconn.send(None)
#         except Exception as e:
#             tb = traceback.format_exc()
#             self._cconn.send((e, tb))
#             # raise e  # You can still rise this exception if you need to

#     @property
#     def exception(self):
#         if self._pconn.poll():
#             self._exception = self._pconn.recv()
#         return self._exception
