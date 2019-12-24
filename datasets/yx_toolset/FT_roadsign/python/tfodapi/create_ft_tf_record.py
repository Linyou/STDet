import argparse
import hashlib
import io
import json
import numpy as np
import os
import PIL.Image
import sys
import tensorflow as tf

import xml.etree.cElementTree as ET
import pycocotools.mask as maskUtils

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
# Patch the location of gfile
tf.gfile = tf.io.gfile
from os.path import join as pj

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, pj(cur_path, '..', '..', '..', '..'))
import yx_toolset.python.utils.data_conversion as data_conversion
from yx_toolset.python.utils.data_conversion import find_det_parent_class

def check_bimask_validity(mask):
    num_pixel = 0
    for p in np.asarray(mask).flatten():
        if p != 0:
            num_pixel += 1
    return num_pixel

def ft_to_tf_example(abs_label_path,
                     abs_img_path,
                     label_map_dict,
                     ann_type):
    label_file = open(abs_label_path, 'r')
    json_dict = json.load(label_file)
    if not json_dict['labeled']:
        return
    if 'object' in json_dict['outputs'].keys():
        # check existence of coi
        contain_coi = False
        for obj in json_dict['outputs']['object']:
            if find_det_parent_class(obj['name']) in label_map_dict.keys():
                contain_coi = True
        if not contain_coi:
            return

    # Comment out to speed up debug runs
    with tf.gfile.GFile(abs_img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    # check image format and validity of the images
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width = int(json_dict['size']['width'])
    height = int(json_dict['size']['height'])
    width_og, height_og = image.size
    if width_og != width or height_og != height:
        print('Inconsistent data: '+ str(width) + ', ' +str(height) + '; ' +
              str(width_og) + ', ' +str(height_og))
        width = width_og
        height = height_og

    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    classes_text = []
    classes = []
    masks = []
    if 'object' in json_dict['outputs'].keys():
        for obj in json_dict['outputs']['object']:
            if ann_type == 'bbox':
                bb_class = find_det_parent_class(obj['name'])
                if bb_class not in label_map_dict.keys():
                    continue
                xmin = float(obj['bndbox']['xmin'])
                ymin = float(obj['bndbox']['ymin'])
                xmax = float(obj['bndbox']['xmax'])
                ymax = float(obj['bndbox']['ymax'])
                if not (xmin >= 0 and ymin >= 0 and xmax < width and ymax < height):
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(width, xmax)
                    ymax = min(height, ymax)
                xmin_list.append(float(xmin) / width)
                ymin_list.append(float(ymin) / height)
                xmax_list.append(float(xmax) / width)
                ymax_list.append(float(ymax) / height)

                classes_text.append(bb_class.encode('utf8'))
                classes.append(label_map_dict[bb_class])
            elif ann_type == 'seg':
                if not 'polygon' in obj.keys(): 
                    continue
                if obj['name'] not in label_map_dict.keys():
                    continue
                res = data_conversion.ft_mask_conversion(width, height, obj)
                if res != None:
                    bb, point_set = res
                    xmin = bb[0]; ymin = bb[1]; xmax = bb[2]; ymax = bb[3]
                    if not (xmin >= 0 and ymin >= 0 and xmax < width and ymax < height):
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(width, xmax)
                        ymax = min(height, ymax)
                    xmin_list.append(float(xmin) / width)
                    ymin_list.append(float(ymin) / height)
                    xmax_list.append(float(xmax) / width)
                    ymax_list.append(float(ymax) / height)

                    rles = maskUtils.frPyObjects([point_set], height, width)
                    rle = maskUtils.merge(rles)
                    mask_np = maskUtils.decode(rle)
                    # mask_remapped = (mask_np != 2).astype(np.uint8)
                    mask_remapped = mask_np.astype(np.uint8)
                    masks.append(mask_remapped)
                    # for row in mask.tolist():
                    #     masks += row

                classes_text.append(obj['name'].encode('utf8'))
                classes.append(label_map_dict[obj['name']])

            
    if len(xmin_list) == 0:
        return

    feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          os.path.basename(abs_img_path).encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          os.path.basename(abs_img_path).encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_list),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_list),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_list),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_list),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }
    if ann_type == 'seg':
        encoded_mask_png_list = []
        for mask in masks:
            # res = check_bimask_validity(mask)
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            # encoded_png = output.getvalue()
            # encoded_png_io = io.BytesIO(encoded_png)
            # mask = np.asarray(PIL.Image.open(encoded_png_io))
            # res = check_bimask_validity(mask)
            encoded_mask_png_list.append(output.getvalue())
        feature['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png_list))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def ftxml_to_ft_example(dirname, 
                           basename,
                           label_map_dict):
    # print(basename)
    tree = ET.parse(pj(dirname, basename))
    root = tree.getroot()

    # check for classes of interest first
    contain_coi = False
    for child in root:
        if child.tag == 'object':
            for entry in child:
                if entry.tag == 'name':
                    if find_det_parent_class(entry.text) in label_map_dict.keys():
                        contain_coi = True
    if not contain_coi:
        return

    # Comment out to speed up debug runs
    img_path = pj(dirname, '..', 'images', basename.split('.')[0] + '.jpg')
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width_og, height_og = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    skip_bb = False
    for child in root:
        if child.tag == 'size':
            for entry in child:
                if entry.tag == 'width':
                    width = int(entry.text)
                    assert width == width_og
                if entry.tag == 'height':
                    height = int(entry.text)
                    assert height == height_og
        if child.tag == 'object':
            for entry in child:
                if entry.tag == 'name':
                    bb_class = find_det_parent_class(entry.text)
                    if not bb_class in label_map_dict.keys():
                        skip_bb = True
                        break
                    classes.append(label_map_dict[bb_class])
                    classes_text.append(bb_class.encode('utf8'))
            if skip_bb:
                skip_bb = False
                continue
            for entry in child:
                if entry.tag == 'bndbox':
                    for coord in entry:
                        if coord.tag == 'xmin':
                            xmin.append(float(coord.text) / width)
                        if coord.tag == 'ymin':
                            ymin.append(float(coord.text) / height)
                        if coord.tag == 'xmax':
                            assert int(coord.text) <= width
                            xmax.append(float(coord.text) / width)
                        if coord.tag == 'ymax':
                            assert int(coord.text) <= height
                            ymax.append(float(coord.text) / height)
            difficult_obj.append(0)
            truncated.append(0)
            poses.append('Unspecified'.encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          os.path.basename(img_path).encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          os.path.basename(img_path).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
      }))
    return example

def convert(data_map, 
            data_map_keys, 
            writer, 
            label_map_dict, 
            ann_type):
    for idx, key in enumerate(data_map_keys):
        print('{}/{} finished...'.format(idx, len(data_map_keys)))
        tf_example = ft_to_tf_example(data_map[key][1],
                                      data_map[key][0],
                                      label_map_dict,
                                      ann_type)
        if tf_example:
            writer.write(tf_example.SerializeToString())

def parse_args():
    parser = argparse.ArgumentParser()
    # set as optional for now
    # better require abs path
    parser.add_argument('--label_map_path', default='/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21/ft_seg_label_map.txt')
    parser.add_argument('--ft_data_dir', default='/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21')
    parser.add_argument('--subfolders', help='Indicates if the input path contains subfolders.',action='store_true')
    parser.add_argument('--label_source', choices=['og', 'xml'], default='og', help='The source of the label files')
    parser.add_argument('--ann_type', choices=['bbox', 'seg'], default='seg')
    parser.add_argument('--train_record_path', default='/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21/train.record')
    parser.add_argument('--valid_record_path', default='/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21/valid.record')
    parser.add_argument('--valid_ratio', default=0.1, help='The ratio of validation files.', type=float)
    return parser.parse_args()

def main(_):
    args = parse_args()

    label_map_dict = label_map_util.get_label_map_dict(args.label_map_path)
    train_writer = tf.io.TFRecordWriter(args.train_record_path)
    valid_writer = tf.io.TFRecordWriter(args.valid_record_path)

    if args.label_source == 'og':
        data_map, shuffled_list, train_size = data_conversion.parse_folder_ft_det(args.ft_data_dir, args.subfolders, args.valid_ratio)
        if args.valid_ratio == 1:
            convert(data_map, shuffled_list, valid_writer, label_map_dict, args.ann_type)
        elif args.valid_ratio == 0:
            convert(data_map, shuffled_list, train_writer, label_map_dict, args.ann_type)
        else:
            convert(data_map, shuffled_list[:train_size], train_writer, label_map_dict, args.ann_type)
            convert(data_map, shuffled_list[train_size:], valid_writer, label_map_dict, args.ann_type)
    
    
    # The fxxml interface needs to be updated   
    # if args.label_source == 'xml':
    #     for file in os.listdir(args.ft_data_dir):
    #         if file.split('.')[-1] != 'xml':
    #             continue
    #         tf_example = ftxml_to_ft_example(args.ft_data_dir,
    #                                             file,
    #                                             label_map_dict)
    #         if tf_example:
    #             writer.write(tf_example.SerializeToString())
    
    train_writer.close()
    valid_writer.close()

if __name__ == '__main__':
    tf.compat.v1.app.run()