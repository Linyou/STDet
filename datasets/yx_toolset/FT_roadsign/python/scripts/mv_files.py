import argparse
import os
import shutil

from os.path import join as pj

def parse_args():
    parser = argparse.ArgumentParser(description='Merge folders')
    parser.add_argument('--prepend', type=bool, default=False, help='Choose whether to prepend the folder name to each file name')
    parser.add_argument('--ft_path')
    parser.add_argument('--out_path', help='Please create \'images\' and \'labels\' folders under it first')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print('Moving files...')
    for folder in os.listdir(args.ft_path):
        if args.prepend:
            prepend = folder + '_'
        else:
            prepend = ''
        folder_path = pj(args.ft_path, folder)
        if not os.path.exists(pj(folder_path, 'labels')):
            continue
        imgs_folder = pj(folder_path, 'images')
        for img in os.listdir(imgs_folder):
            shutil.copy(pj(imgs_folder, img), pj(args.out_path, 'images', prepend + img))
        labels_folder = pj(folder_path, 'labels')
        for label in os.listdir(labels_folder):
            shutil.copy(pj(labels_folder, label), pj(args.out_path, 'labels', prepend + label))