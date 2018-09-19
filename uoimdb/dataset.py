import os
import cv2
import glob
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
from .utils import cache_result
from . import config as cfg



@cache_result(lambda path: 'cache/cached-' + os.path.basename(path) + '.pkl')
def get_ground_truth(path):
    df = pd.read_csv(path)
    
    # remove accidental boxes
    df = df[(df.w != 0) & (df.h != 0)]
    
    # get box corners
    df.loc[:,'x1'] = (df.x - df.w/2)
    df.loc[:,'y1'] = (df.y - df.h/2)
    df.loc[:,'x2'] = (df.x + df.w/2)
    df.loc[:,'y2'] = (df.y + df.h/2)
    
    # map plume origins
    only_plumes = df.label == 'plume'
    if 'origin_id' not in df.columns:
        print('getting plume origins...')
        df.loc[:,'origin_id'] = None
        df.loc[only_plumes] = df[only_plumes].apply(get_origin_id, axis=1, df=df[only_plumes])

    # get the duration of each plume (in frames)
    if 'plume_dur' not in df.columns:
        print('getting plume durations...') # get the frame duration for each plume
        plume_dur = df[only_plumes].groupby('origin_id').src.count()
        df.loc[only_plumes, 'plume_dur'] = plume_dur[df[only_plumes].origin_id].values
    
    # get idx from src
    df['idx'] = df.src.apply(lambda x: x.replace('/', ','))
    return df.set_index('idx')



class Dataset(object):
    classes = ['__background__', 'plume', 'shadow', 'cloud', 'light', 'ambiguous']
    
    def __init__(self, name='UOImages-full'):
        self.name = name
        self.idxs = list({os.path.splitext(os.path.basename(path))[0] 
                    for path in glob.glob(self.img_path.format('*'))})
        
        self.h, self.w, self.depth = cv2.imread(self.img_path.format(self.idxs[0])).shape
        
    def __str__(self):
        return '''<{}: {}. {} images ({}x{}x{}). 
        classes: {}>'''.format(
            self.__class__.__name__, self.name, len(self.idxs), 
            self.h, self.w, self.depth, self.classes)
    
    def __repr__(self):
        return self.__str__()
    
    def get_annotation(self, idx):
        raise NotImplemented
    
    def get_boxes(self, idx):
        raise NotImplemented
    
    def get_image(self, idx):
        return 255 - cv2.imread(self.img_path.format(idx))
    
    def get_detections(self, path, thresh=None, **kw):

        # the nested function is so that we can cache the entire dataframe and then apply the threshold after loading.
        @cache_result(lambda: 'cache/processed-'+os.path.basename(path))
        def get_dets():
            with open(path, 'rb') as f:
                detections = pickle.load(f, encoding='latin1')

            df = pd.DataFrame([
                [self.classes[i], self.idxs[j], k] + det.tolist()
                for i, d_cls in enumerate(detections)
                for j, d_img in enumerate(d_cls)
                for k, det in enumerate(d_img)
            ], columns=['label', 'idx', 'i', 'x1', 'y1', 'x2', 'y2', 'score']).set_index('idx')#['idx', 'i']

            df = transform_boxes_scale(df, in_scale=(self.h, self.w)) # normalize between 0, 1
            df = transform_boxes_crop(df, in_crop=True) # undo crop
            return df

        df = get_dets(**kw)
        if thresh is not None:
            df = df[df.score >= thresh]
        
        return df
    
#     @cache_result(lambda self, path: 'cache/cached-' + os.path.basename(path) + '.pkl')
#     def get_ground_truth_annotations(self, path):
#         df = pd.DataFrame([
#             (label, idx, x1, y1, x2, y2)
#             for idx in self.idxs
#             for label, (x1, y1), (x2, y2) in self.get_boxes(idx, norm=True)
#         ], columns=['label', 'idx', 'x1', 'y1', 'x2', 'y2']).set_index('idx')
        
#         transform_boxes_crop(df, in_crop=True) # undo crop
#         return df



class VOCDataset(Dataset):
    '''Loads in data output by build_dataset.py'''
    
    def __init__(self, name='UOImages-full'):
        self.base_dir = os.path.join('Data', name)
        self.img_path = os.path.join(self.base_dir, 'Images/{}.jpg')
        self.ann_path = os.path.join(self.base_dir, 'Annotations/{}.xml')
        self.split_path = os.path.join(self.base_dir, 'ImageSets/{}/{}.txt')
        Dataset.__init__(self, name)
        
    
    def get_annotation(self, idx):
        return ET.parse(self.ann_path.format(idx))
    
    def get_boxes(self, idx, norm=False):
        tree = self.get_annotation(idx)
        if norm:
            shape = tree.find('size')
            h, w = shape.find('H') or self.h, shape.find('W') or self.w
        
        for obj in tree.findall('object'):
            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            
            label = obj.find('name').text.lower().strip()
            if norm:
                pt1, pt2 = (x1/w, y1/h), (x2/w, y2/h)
            else:
                pt1, pt2 = (x1, y1), (x2, y2)
            yield label, pt1, pt2
            
    def get_shape(self, idx):
        tree = self.get_annotation(idx)
        shape = tree.find('size')
        return shape.find('H'), shape.find('W')
                  
    def get_split(self, split, section='Main'):
        with open(self.split_path.format(section, split), 'r') as f:
            return [l.strip() for l in f.readlines()]
    

    
    
    
def get_origin_id(row, df, fill_all=True):
    id = row.name
    while True:
        try: # get the previous id of previous id
            prev_id = df.loc[id, 'prev_id']
        except (KeyError, TypeError) as e:
            break # id is missing, broken link
        if pd.isnull(prev_id):
            break # no previous box, chain ended
        id = prev_id
    if fill_all or id != row.name:
        row['origin_id'] = id
    return row


def transform_boxes_crop(boxes, in_crop=None, out_crop=None):
    boxes = boxes.copy() # stop SettingWithCopyWarning -.-
    def_crop = (cfg.CROP_X1 or 0, cfg.CROP_Y1 or 0), (cfg.CROP_X2 or 1, cfg.CROP_Y2 or 1)
    if in_crop is not None:
        (x1, y1), (x2, y2) = in_crop if in_crop is not True else def_crop
        # print((x1, y1), (x2, y2))
        boxes.x1 = boxes.x1 * (x2 - x1) + x1
        boxes.x2 = boxes.x2 * (x2 - x1) + x1
        boxes.y1 = boxes.y1 * (y2 - y1) + y1
        boxes.y2 = boxes.y2 * (y2 - y1) + y1
    if out_crop is not None:
        (x1, y1), (x2, y2) = out_crop if out_crop is not True else def_crop
        boxes.x1 = (boxes.x1 - x1) / (x2 - x1)
        boxes.x2 = (boxes.x2 - x1) / (x2 - x1)
        boxes.y1 = (boxes.y1 - y1) / (y2 - y1)
        boxes.y2 = (boxes.y2 - y1) / (y2 - y1)
    return boxes


def transform_boxes_scale(boxes, in_scale=None, out_scale=None):
    boxes = boxes.copy() # stop SettingWithCopyWarning -.-
    if in_scale is not None:
        h, w = in_scale if in_scale is not True else (cfg.RESCALE, cfg.RESCALE)
        boxes.x1 /= w
        boxes.x2 /= w
        boxes.y1 /= h
        boxes.y2 /= h
    elif out_scale is not None:
        h, w = out_scale if out_scale is not True else (cfg.RESCALE, cfg.RESCALE)
        boxes.x1 *= w
        boxes.x2 *= w
        boxes.y1 *= h
        boxes.y2 *= h
    return boxes