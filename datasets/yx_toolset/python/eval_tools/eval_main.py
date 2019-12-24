import argparse
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.io as io
import sys
import pylab
# pylab.rcParams['figure.figsize'] = (12.0, 15.0)

from collections import defaultdict
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from os.path import join as pj
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(annFile, resFile, annType, per_cls_stat=False):
    """
        Args: annType has the following types ['segm','bbox','keypoints']

    """
    res = []
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    res.append(cocoEval.stats[1])
    
    catIds = cocoGt.getCatIds()
    cats = cocoGt.loadCats(catIds)
    for cat in cats:
        print('\nClass ' + cat['name'] + ' stats:')
        cocoEval.params.catIds = [cat['id']]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        res.append(cocoEval.stats[1])
    cats = [{'name': 'Overall'}] + cats
    for idx, cls_ in enumerate(cats):
        print('{} mAP@.5: {:0.3f}'.format(cats[idx]['name'], res[idx]))

def evaluate_curve(annFile, resFile, annType, score_thr=0.2, split=10):
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    anns = cocoDt.dataset['annotations']
    anns = [ann for ann in anns if ann['score'] >= score_thr]

    # cocoDt = cocoGt.loadRes(anns)
    # cocoEval = COCOeval(cocoGt,cocoDt,annType)
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    # for i in range(0, 100, 2):
    #     pr = np.mean(cocoEval.eval['precision'][0,i,:,0,2])
    #     print('ACC: {:.3f}, RECALL: {:.3f}'.format(pr, (i + 1) / 100.0))
    # prs = [np.mean(cocoEval.eval['precision'][0,i,:,0,2]) for i in range(0, 101)]
    # x = np.arange(0.0, 1.01, 0.01)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.xlim(0, 1.0)
    # plt.ylim(0, 1.01)
    # plt.grid(True)
    # plt.plot(x, prs)
    # plt.show()

    anns = sorted(anns, key=lambda ann: -ann['score'])
    n = int(len(anns) / split) # chunk length
    anns_chunks = [anns[0:i + n] for i in range(0, len(anns), n)]
    num_gt = len(cocoGt.anns)
    output = []
    for chunk in anns_chunks:
        tmp_json_dict = {'images': copy.deepcopy(cocoGt.dataset['images']),
                         'categories':  copy.deepcopy(cocoGt.dataset['categories']),
                         'annotations': chunk}
        with open('tmp.json', 'w') as outfile:
            json.dump(tmp_json_dict, outfile)
        cocoDt = COCO('tmp.json')
        cocoDt.dataset['annotations'] = chunk
        cocoDt.createIndex()
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.evaluate()

        num_hit = 0
        K0 = len(cocoEval.params.catIds)
        A0 = len(cocoEval._paramsEval.areaRng)
        I0 = len(cocoEval._paramsEval.imgIds)
        res_list = [cocoEval.evalImgs[k * A0 * I0 + a * I0 + i] 
                    for i in range(0, I0)
                    for a in [1, 2, 3]
                    for k in range(0, K0)]
        res_list = [res for res in res_list if not res is None]
        for res in res_list:
            num_hit += np.count_nonzero(res['gtMatches'][0]) # .5 IOU hits
        num_hit /= 3
        output.append((num_hit / len(chunk), num_hit / num_gt))
    for out in output:
        print('ACC: {:.3f}, RECALL: {:.3f}'.format(out[0], out[1]))

        # E = [e for e in cocoEval.evalImgs if not e is None]
        # dtScores = np.concatenate([e['dtScores'][0:100] for e in E])
        # inds = np.argsort(-dtScores, kind='mergesort')
        # dtm  = np.concatenate([e['dtMatches'][:,0:100] for e in E], axis=1)[:,inds]
        # dtIg = np.concatenate([e['dtIgnore'][:,0:100]  for e in E], axis=1)[:,inds]
        # tps = np.logical_and(dtm, np.logical_not(dtIg))
        # tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
        # tp = tp_sum[0]
        # print('ACC: {:.3f}, RECALL: {:.3f}'.format(tp / len(chunk), tp / num_gt))

def showBndbox(coco, anns):
    """
        Args: Dataset object in coco format

    """
    ax = plt.gca()
    rectangles = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        if 'bbox' in ann:
            # score = float(ann['score'])
            # if score < 0.5: continue
            bb = ann['bbox']
            rectangles.append(Rectangle((bb[0], bb[1]), bb[2], bb[3]))
            color.append(c)
            text = coco.loadCats(ann['category_id'])[0]['name']
            # if 'score' in ann:
            #   text = '%s: %.2f' % (coco.loadCats(ann['category_id'])[0]['name'], score)
            # else:
            #   text = coco.loadCats(ann['category_id'])[0]['name']
            ax.text(bb[0], bb[1] - 15, text, color=[1, 1, 1], backgroundcolor=c, weight='bold')
    p = PatchCollection(rectangles, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)

def coco_format_viz(img_folder, annFile):
    coco = COCO(annFile)

    cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]
    # catIds = coco.getCatIds(catNms=[nms[np.random.randint(0, len(nms))]])
    all_cat_id = [cat['id'] for cat in cats]
    catIds = coco.getCatIds(catNms=all_cat_id)
    imgIds = coco.getImgIds(catIds=catIds)
    for _ in range(50):
        img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        I = io.imread(pj(img_folder, img['file_name']))
        plt.imshow(I); plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        # showBndbox(coco, anns)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.show()

    # catIds = coco.getCatIds()
    # imgIds = coco.getImgIds()
    # # catIds = [0, 1, 20]

    # for imgId in random.sample(imgIds, 50):
    #     img = coco.loadImgs(imgId)[0]
    #     I = io.imread(pj(img_folder, img['file_name']))

    #     plt.imshow(I); plt.axis('off')
    #     annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    #     anns = coco.loadAnns(annIds)
    #     print('Height: {}\n Width: {}\n'.format(img['height'], img['width']))
    #     showBndbox(coco, anns)
    #     fig = plt.gcf()
    #     fig.set_size_inches(18.5, 10.5)
    #     plt.title(img['file_name'])
    #     # plt.savefig(pj('/home/yingges/Desktop/yingges/experiments/data/ft_det_cleanedup/ignore_toosmall/11_30/img_annoed', img['file_name']))
    #     plt.show()
    #     plt.close()

def data_stats(annFile, categories):
    with open(annFile) as file:
        json_dict = json.load(file)
    # Collect stats on the shorter side of the bbs
    size_list = []
    cls_cnt = defaultdict(int)
    for ann in json_dict['annotations']:
        cls_name = categories[ann['category_id'] - 1]
        size_list.append(min(ann['bbox'][2], ann['bbox'][3]))
        # if cls_name in ['rn', 'ro', 'lo', 'ors']:
        #     cls_name = 'panel'
        # elif cls_name.startswith('p'):
        if cls_name.startswith('p'):
            cls_name = 'po'
        cls_cnt[cls_name] += 1
    total_sample_cnt = 0
    for k, v in cls_cnt.items():
        total_sample_cnt += v
    cls_percent = dict()
    for k, v in cls_cnt.items():
        cls_percent[k] = v / total_sample_cnt
    print(cls_cnt)
    print(cls_percent)
    plt.hist(size_list, bins=20)
    plt.show()

PREDEFINED_CLASSES_GENERIC = ['i','p', 'wo', 'rn', 'lo', 'tl',  'ro']
PREDEFINED_CLASSES_GENERIC_UPDATED_TEST = ['i', 'p', 'wo', 'rn', 'lo', 
                                           'tl', 'ro', 'sc0', 'sc1']
PREDEFINED_CLASSES_GENERIC_UPDATED_TRAIN = ['i', 'p', 'wo', 'rn', 'lo', 
                                            'tl', 'ro', 'sc0', 'sc1', 'ors']
PREDEFINED_CLASSES = ['io', 'wo', 'ors', 'p10', 'p11', 
                      'p26', 'p20', 'p23', 'p19', 'pne',
                      'rn', 'ps', 'p5', 'lo', 'tl',
                      'pg', 'sc1','sc0', 'ro', 'pn',
                      'po', 'pl', 'pm']

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', default='eval', choices=['eval', 'viz', 'stats'])
    parser.add_argument('--ann_type', default='bbox', choices=['segm','bbox','keypoints'])
    parser.add_argument('--anno_file_path', default='/media/yingges/Data/Datasets/COCO/annotations/instances_val2017.json', type=str)
    parser.add_argument('--img_folder_path', default='/media/yingges/Data/Datasets/COCO/val2017',type=str)
    parser.add_argument('--res_file_path', default='/media/yingges/Data/201910/yolact/yolact/results/mask_detections.json')
    parser.add_argument('--per_cls_stat', action='store_true')
    parser.add_argument('--map_curve', default=False, action='store_true')
    parser.add_argument('--score_thr', default=0, help='This only works when evaluating the global average precision.', type=float)
    parser.add_argument('--finegrained_cls', default=False, action='store_true')
    args = parser.parse_args()

    # categories = PREDEFINED_CLASSES if args.finegrained_cls else PREDEFINED_CLASSES_GENERIC_UPDATED_TEST
    categories = PREDEFINED_CLASSES if args.finegrained_cls else PREDEFINED_CLASSES_GENERIC_UPDATED_TRAIN
    if args.mode == 'viz':
        coco_format_viz(args.img_folder_path, args.anno_file_path)
    elif args.mode == 'eval':
        if args.map_curve:
            evaluate_curve(args.anno_file_path, args.res_file_path, args.ann_type, args.score_thr)
        else:
            evaluate(args.anno_file_path, args.res_file_path, args.ann_type, args.per_cls_stat)
        data_stats(args.anno_file_path, categories)
    elif args.mode =='stats':
        data_stats(args.anno_file_path, categories)

if __name__ == '__main__':
    main()