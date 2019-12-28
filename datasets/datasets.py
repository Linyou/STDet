# -*- coding: utf-8 -*-



from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


def get_metadata():
    meta = {
        "thing_classes": ['io', 'lo', 'tl', 'rn', 'ro', 'wo', 'ors', 'sc1',
                          'sc0', 'p26', 'p20', 'p23', 'ps', 'pne', 'pg',
                          'pn', 'po', 'pl', 'pm', 'p10', 'p11', 'p19', 'p5'],
    }
    return meta


SPLITS = {
    "sfdet_train": ("/home/youtian/data/ft_det/ft_det_cleanedup/images", 
                    "/home/youtian/data/ft_det/ft_det_cleanedup/train.json"),
    "sfdet_val": ("/home/youtian/data/ft_det/ft_det_cleanedup/images", 
                  "/home/youtian/data/ft_det/ft_det_cleanedup/valid.json"),
}



for key, (image_root, json_file) in SPLITS.items():
    # Assume pre-defined datasets live in `./datasets`.

    DatasetCatalog.register(
        key,
        lambda key=key, json_file=json_file, image_root=image_root: load_coco_json(
            json_file, image_root, key
        ),
    )

    MetadataCatalog.get(key).set(
        json_file=json_file, image_root=image_root, **get_metadata()
    )
