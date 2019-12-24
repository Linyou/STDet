import argparse
import errno    
import os
import random

from os.path import join as pj
from shutil import copy

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_merged_dir', default ='/media/yingges/Data/201910/FT/FTData/ft_od1_merged')
    parser.add_argument('--out_dir', default='/media/yingges/Data/201910/FT/FTData/ft_od1_merged/anno_fixing')
    parser.add_argument('--split_num', default=2)
    args = parser.parse_args()

    fname_txt_list = []
    for i in range(args.split_num):
        mkdir_p(pj(args.out_dir, str(i), 'images'))
        mkdir_p(pj(args.out_dir, str(i), 'labels'))
        fname_txt_list.append(open(pj(args.out_dir, str(i), 'fname_list.txt'), 'w+'))

    ftfilelist = [f.split('.')[0] for f in os.listdir(pj(args.ft_merged_dir, 'labels')) if f.endswith('.json')]
    random.shuffle(ftfilelist)

    for i in range(len(ftfilelist)):
        set_id = i % args.split_num
        assert os.path.exists(pj(args.ft_merged_dir, 'images', ftfilelist[i] + '.jpg'))
        copy(pj(args.ft_merged_dir, 'images', ftfilelist[i] + '.jpg'), pj(args.out_dir, str(set_id), 'images'))
        copy(pj(args.ft_merged_dir, 'labels', ftfilelist[i] + '.json'), pj(args.out_dir, str(set_id), 'labels'))
        fname_txt_list[set_id].write(ftfilelist[i] + '\n')

    for f in fname_txt_list:
        f.close()
