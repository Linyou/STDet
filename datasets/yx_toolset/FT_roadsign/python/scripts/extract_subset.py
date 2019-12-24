import json
import os
import random
import shutil

from os.path import join as pj

def extract_subset(folder_path, output_path, size=100):
    all_f_names = [f.split('.')[0] for f in os.listdir(pj(folder_path, 'images'))]
    random.shuffle(all_f_names)
    sample_files = all_f_names[:size]
    img_files = [pj(folder_path, 'images', f + '.jpg') for f in sample_files]
    anno_files = [pj(folder_path, 'labels', f + '.json') for f in sample_files]
    for img, anno in zip(img_files, anno_files):
        shutil.copy(img, pj(output_path, 'images'))
        shutil.copy(anno, pj(output_path, 'labels'))

def extract_splitset(json_file, data_path, output_path):
    json_dict = json.load(open(json_file))
    for im in json_dict['images']:
        shutil.copy(pj(data_path, 'images', im['file_name']), pj(output_path, 'images'))
        shutil.copy(pj(data_path, 'labels', im['file_name'].split('.')[0] + '.json'), pj(output_path, 'labels'))

if __name__ == '__main__':
    # extract_subset('/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21',
    #              '/home/neut/Desktop/yingges/201911/data/yunxikeji-01-2019-10-21/sample20',
    #              20)
    extract_splitset('/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/valid.json',
                     '/media/yingges/Data/201910/FT/FTData/ft_od1_merged',
                     '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/og_files')