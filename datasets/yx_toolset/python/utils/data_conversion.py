import json
import math
import os
import random

from glob import glob
from os.path import join as pj

def add_item(in_list, out_dict):
    for item in in_list:
        try:
            out_dict[os.path.basename(item).split('.')[0]].append(item)
        except:
            out_dict[os.path.basename(item).split('.')[0]] = [item]

def parse_folder_ft_det(input_path, subfolders, valid_ratio):
    """
    Point of having this as a separate interface is that
    we can easily deal with changes of file structure.

    Args:


    Returns:
        res_dict: {data_key: [abs_img_path, abs_label_path], ...}

    """
    if subfolders:
        folder_list = [item for item in os.listdir(input_path) if os.path.isdir(pj(input_path, item))]
        img_list = []
        anno_list = []
        for folder in folder_list:
            img_list += glob(pj(input_path, folder, 'images', '*')) # TODO: file type check
            anno_list += glob(pj(input_path, folder, 'labels', '*'))
    else:
        img_list = glob(pj(input_path, 'images', '*'))
        anno_list = glob(pj(input_path, 'labels', '*'))
    img_list = [im for im in img_list if im.endswith('.jpg')]
    anno_list = [ann for ann in anno_list if ann.endswith('.json')]
    # returns a dict where each entry is a list: [abspath_to_img, abspath_to_anno]
    res_dict = {}
    add_item(img_list, res_dict)
    add_item(anno_list, res_dict)
    for k, v in res_dict.items():
        # remove that from the dict if one file(img/ann) is missing
        if len(v) == 1:
            del res_dict[k]

    shuffled_list = list(res_dict.keys())
    random.shuffle(shuffled_list)
    train_size = int(math.floor(len(shuffled_list) * (1 - float(valid_ratio))))

    return res_dict, shuffled_list, train_size

def find_det_parent_class(in_cls, finegrained_cls):
    if finegrained_cls:
        if in_cls.startswith('pl'):
            return 'pl'
        elif in_cls.startswith('pm'):
            return 'pm'
        elif in_cls.startswith('ph'):
            return 'ph'
        else:
            return in_cls
    else:
        if in_cls.startswith('p'):
            return 'p'
        if in_cls.startswith('i'):
            return 'i'
        return in_cls

def ft_mask_conversion(width, height, anno):
    xmin = width
    ymin = height
    xmax = 0
    ymax = 0
    point_set = []
    # if anno is a dictionary
    for i in range(int(len(anno['polygon']) / 2)):
        x = anno['polygon']['x' + str(i + 1)]
        y = anno['polygon']['y' + str(i + 1)]
        if(x < 0):
            x = 0
        if(x > width):
            x = width
        if(y < 0):
            y = 0
        if (y > height):
            y = height
        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x, xmax)
        ymax = max(y, ymax)
        point_set.append(x)
        point_set.append(y)

    # A simple check to detect the illegal segementation annotation
    if len(point_set) <= 4:
        print('Illegal segmentation annotation!')
        return

    return [xmin, ymin, xmax, ymax], point_set

# PREDEFINED_CLASSES_GENERIC = ['i','p', 'wo', 'rn', 'lo', 'tl',  'ro']
PRE_DEFINE_CATEGORIES_GENERIC = {'i': 1, 'p': 2, 'wo': 3, 'rn': 4, 'lo': 5, 
                                 'tl': 6, 'ro': 7}
PREDEFINED_CLASSES = ['io', 'wo', 'ors', 'p10', 'p11', 
                      'p26', 'p20', 'p23', 'p19', 'pne',
                      'rn', 'ps', 'p5', 'lo', 'tl',
                      'pg', 'sc1','sc0', 'ro', 'pn',
                      'po', 'pl', 'pm']

def merge_cls(inFile, outFile):
    with open(inFile) as f:
        json_list = json.load(f)
    out_json_list = []
    for ann in json_list:
        if ann['category_name'] in ['ors', 'sc0', 'sc1']:
            continue
        elif ann['category_name'].startswith('p'):
            ann['category_name'] = 'p'
            ann['category_id'] = PRE_DEFINE_CATEGORIES_GENERIC['p']
        elif ann['category_name'] == 'io':
            ann['category_name'] = 'i'
            ann['category_id'] = PRE_DEFINE_CATEGORIES_GENERIC['i']
        else:
            ann['category_id'] = PRE_DEFINE_CATEGORIES_GENERIC[ann['category_name']]
        out_json_list.append(ann)
    with open(outFile, 'w') as f:
        f.write(json.dumps(out_json_list))

if __name__ == '__main__':
    merge_cls('/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/og_files/generic_valid_sizethr625_fpn.json',
              '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/og_files/merged_generic_valid_sizethr625_fpn.json')