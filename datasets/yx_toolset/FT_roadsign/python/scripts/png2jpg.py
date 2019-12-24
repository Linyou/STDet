import argparse
import cv2
import numpy as np
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert png images to jpeg ones')
    parser.add_argument('--in_path')
    path = parser.parse_args().in_path

    for filename in os.listdir(path):
        
        if os.path.splitext(filename)[1] == '.png':
            # print(filename)
            img = cv2.imread(path + filename)
            # print(filename.replace(".png",".jpg"))
            newfilename = filename.replace(".png",".jpg")
            ret = cv2.imwrite(path + newfilename, img)
            if not ret:
                print(filename)
                os.remove(path + newfilename)

