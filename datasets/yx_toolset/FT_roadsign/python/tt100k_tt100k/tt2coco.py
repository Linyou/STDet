import argparse
import json
import os

from os.path import join as pj

def main():
	parser = argparse.ArgumentParser('')
	parser.add_argument('--input_path', default='/media/yingges/Data/Datasets/TT100K_TS/data')
	args = parser.parse_args()

	data_path = args.input_path
	tt_json = json.load(open(pj(data_path, 'annotations.json')))

if __name__ == '__main__':
	main()