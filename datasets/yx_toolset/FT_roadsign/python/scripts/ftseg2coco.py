# Conver FT segmentation data into COCO format.
# The bounding rectangles of the polygon annotations are considered the bounding boxes.
#
# TODO: Add COCO polygon verification

import cv2
import json
import math
import os
import random
import re

from os.path import join as pj

# PRE_DEFINE_CATEGORIES = None
# PRE_DEFINE_CATEGORIES = {'roa': 1, 'loa': 2, 'soa': 3, 'sloa': 4, 'sroa': 5,
#                          'ooa': 6, 'cf': 7, 'rg': 8, 'np': 9, 'cross': 10}
# PRE_DEFINE_CATEGORIES = {'roa': 1, 'loa': 2, 'soa': 3, 'sloa': 4, 'sroa': 5,
#                          'ooa': 6, 'cf': 7, 'rg': 8, 'np': 9, 'cross': 10,
#                          'ld': 11,'zyfgd': 12,'lcfgd': 13,'lmj': 14,'sfwl': 15,
#                          'sdwl': 16,'sfyl': 17,'sdyl': 18,'dfyl': 19,'sl': 20}
PRE_DEFINE_CATEGORIES = {'roa': 1, 'loa': 2, 'soa': 3, 'sloa': 4, 'sroa': 5,
                         'yjg': 6, 'cf': 7, 'rg': 8, 'np': 9, 'cross': 10,
                         'ld': 11,'zyfgd': 12,'lcfgd': 13,'lmj': 14,'sfwl': 15,
                         'sdwl': 16,'sfyl': 17,'sdyl': 18,'dfyl': 19,'sl': 20}
broken_f_cnt = 0
illegal_anno_cnt = 0

def get_categories(ft_files):
    classes_names = set()
    for filename in ft_files:
        file = open(filename)
        json_dict = json.load(file)
        for top_tuple in json_dict.keys():

            if top_tuple == 'outputs':
                # print out the images that don't have annotation
                # if 'object' not in json_dict['outputs'].keys():
                #     print(filename + ' contains no object?')
                for output in json_dict['outputs'].keys():

                    if output == 'object':
                        for bb in json_dict['outputs']['object']:
                            classes_names.add(bb['name'])
            
    return {name: i for i, name in enumerate(classes_names)}

def write_one_image(json_dict, out_json_dict, categories, anno_id, full_fname, img_id):

    jpg_dir = pj(os.path.dirname(full_fname), '..', 'images')
    img_name = os.path.basename(full_fname).split('.')[0] + '.jpg'
    # Check if this image contains class of interest. If not, skip it
    contain_coi = False
    for anno in json_dict['outputs']['object']:
        if anno['name'] in categories:
            if 'polygon' in anno.keys(): # rule out the bb annotation
                contain_coi = True
    if not contain_coi:
        return

    img = cv2.imread(pj(jpg_dir , img_name))
    if img is None:
        global broken_f_cnt
        broken_f_cnt += 1
        return
    height, width, _ = img.shape
    # img_id = int( img_name.split('COCO_test2015_')[1])
    img_id += 1
    img_info = {
        'file_name': img_name,
        'height': height,
        'width': width,
        'id': img_id
    }

    # paste this whenever you want to visualize the json record
    # print(json.dumps(json_dict, indent=4))
    valid_anno = 0
    for anno in json_dict['outputs']['object']:
        if anno['name'] in categories.keys() and 'polygon' in anno.keys():
            cate_id = categories[anno['name']]

            xmin = width
            ymin = height
            xmax = 0
            ymax = 0
            point_set = []
            for i in range(int(len(anno['polygon']) / 2)):
                x = anno['polygon']['x' + str(i + 1)]
                y = anno['polygon']['y' + str(i + 1)]
                if(x <0) :
                    x = 0
                if(x > width ):
                    x = width
                if(y < 0):
                    y = 0
                if (y>height):
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
                global illegal_anno_cnt
                illegal_anno_cnt += 1
                continue
            
            bb = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
            area = bb[2] * bb[3]
            anno_entry = {
                'area': area,
                'bbox': bb,
                'iscrowd': 0,
                'image_id': img_id,
                'category_id': cate_id,
                'id': anno_id,
                'ignore': 0,
                'segmentation': [point_set],
            }

            # coco.annToMask(anno_entry)
            # try:
            #     coco.annToMask(anno_entry)
            # except:
            #     global illegal_anno_cnt
            #     illegal_anno_cnt += 1
            #     print('Illegal segmentation annotation!')
            #     print(anno_entry)


            out_json_dict['annotations'].append(anno_entry)
            valid_anno += 1
            anno_id += 1

    if valid_anno == 0:
        return

    out_json_dict['images'].append(img_info)

    return  img_id, anno_id

def convert(ft_files, json_file):
    # check for filename in the image folder?
    out_json_dict = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(ft_files)

    img_id = 1
    anno_id = 1
    for fname in ft_files:
        file = open(fname)
        json_dict = json.load(file)
        for top_tuple in json_dict.keys():

            if top_tuple == 'outputs':
                for output in json_dict['outputs'].keys():

                    # This is equivalent to checking json['labeled'] but is more generalized
                    if output == 'object':
                        ret = write_one_image(json_dict, out_json_dict, categories, anno_id, fname, img_id)
                        if ret:
                            img_id, anno_id = ret


    for cate, cid in categories.items():
        cate_entry = {'supercategory': 'none', 'id': cid, 'name': cate}
        out_json_dict['categories'].append(cate_entry)
    # sort it in the out dict
    out_json_dict['categories'].sort(key=lambda val: val['id'])

    out_file = open(json_file, 'w')
    out_file.write(json.dumps(out_json_dict))
    out_file.close()

if __name__ == '__main__':
    import argparse

    cur_file_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser('')
    parser.add_argument('--ft_dir', default='/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21', type=str)
    parser.add_argument('--train_json_file', default='/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21/train.json', type=str)
    parser.add_argument('--valid_json_file', default='/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21/valid.json', type=str)
    parser.add_argument('--valid_ratio', default=0.15, type=float)
    args = parser.parse_args()

    # add os.path.dirname(os.path.abspath(__file__)), 
    # or restrict the running dir of this file
    ftfilelist = [pj(args.ft_dir, 'labels', item) for item in os.listdir(pj(args.ft_dir, 'labels')) if item.endswith('.json')]
    random.shuffle(ftfilelist)
    train_size = int(math.floor(len(ftfilelist) * (1 - args.valid_ratio)))
    if args.valid_ratio == 1:
        convert(ftfilelist, args.valid_json_file)
    else:
        convert(ftfilelist[:train_size], args.train_json_file)
        convert(ftfilelist[train_size:], args.valid_json_file)
    global broken_f_cnt, illegal_anno_cnt
    print('# of broken images: ' + str(broken_f_cnt))
    print('# of illegal segmentation annotations: ' + str(illegal_anno_cnt))
