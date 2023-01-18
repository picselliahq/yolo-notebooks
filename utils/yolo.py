import numpy as np
from pycocotools.coco import COCO
import os 
from enum import Enum
from typing import List
import logging
import tqdm
import yaml
import glob 
from picsellia.types.enums import InferenceType


class YOLOv(Enum):
    V8 = "V8"
    V7 = "V7"
    V5 = "V5"


class YOLOFormatter:
    def __init__(self, fpath: str, imdir: str, mode: InferenceType, steps = ["train", "test", "val"]) -> None:
        """ 
            fpath (str): path to COCO .json file
            imdir (str): path to your images folder
            targetdir (str): path the target dir for the final YOLO formatted dataset.
            mode (InferenceType): "OBJECT_DETECTION", "SEGMENTATION", "CLASSIFICATION"        
        """
        self.fpath = fpath
        self.imdir = imdir
        self.mode = mode 
        self.steps = [steps] if isinstance(steps, str) else steps

    def __countList(self, lst1, lst2):
        return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]

    def _coco_poly2yolo_poly(self, ann, im_w, im_h) -> List[float]:
        pair_index = np.arange(0, len(ann), 2)
        impair_index = np.arange(1, len(ann), 2)
        Xs = list(map(ann.__getitem__, pair_index))
        xs = list(map(lambda x: x/im_w, Xs))
        Ys = list(map(ann.__getitem__, impair_index))
        ys = list(map(lambda x: x/im_h, Ys))
        return self.__countList(xs, ys)

    def _coco_bbox2yolo_bbox(self, ann, im_w, im_h) -> List[float]:
        x1, y1, w, h = ann["bbox"]
        return [((2*x1 + w)/(2*im_w)) , ((2*y1 + h)/(2*im_h)), w/im_w, h/im_h]

    def _coco_classif2yolo_classif(self, ann, im_w, im_h):
        return []


    def coco2yolo(self, ann, im_w, im_h) -> callable:
        if self.mode == InferenceType.OBJECT_DETECTION:
            return self._coco_bbox2yolo_bbox(ann, im_w, im_h)
        elif self.mode == InferenceType.SEGMENTATION:
            return self._coco_poly2yolo_poly(ann, im_w, im_h)
        elif self.mode == InferenceType.CLASSIFICATION:
            return self._coco_classif2yolo_classif(ann, im_w, im_h)

    def convert(self):
        assert os.path.isdir(os.path.join(self.imdir, "train")), "you must put your images under train/test/val folders."
        assert os.path.isdir(os.path.join(self.imdir, "test")), "you must put your images under train/test/val folders."
        assert os.path.isdir(os.path.join(self.imdir, "val")), "you must put your images under train/test/val folders."
        
        for split in ['train', 'test', 'val']:
            self.coco = COCO(self.fpath)
            logging.info(f"Formatting {split} folder ..")
            dataset_path = os.path.join(self.imdir, split)
            image_filenames = os.listdir(os.path.join(dataset_path, 'images'))
            labels_path = os.path.join(dataset_path, 'labels')
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            for img in tqdm.tqdm(self.coco.loadImgs(self.coco.imgs)):
                result = []
                if img["file_name"] in image_filenames : # check if image is inside your folder first
                    txt_name = img['file_name'][:-4] + '.txt'
                    for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img['id'])):
                        line = " ".join([str(x) for x in  self.coco2yolo(ann, img['width'], img['height'])])
                        result.append(f"{ann['category_id']} {line}")
                    with open(os.path.join(labels_path, txt_name), 'w') as f:
                        f.write("\n".join(result))

    def generate_yaml(self, dpath: str = "data.yaml") -> str:
        names = [label["name"] for label in self.coco.loadCats(self.coco.cats)]
        data_config = {
            'train' : os.path.join(self.imdir, 'train'),
            'val' : os.path.join(self.imdir, 'val'),
            'test' : os.path.join(self.imdir, 'test'),
            'nc' : len(names),
            'names' : names
        }
        f = open(dpath, 'w+')
        yaml.dump(data_config, f, allow_unicode=True)
        return dpath

def get_latest_file(path, run_type: InferenceType):
    if run_type == InferenceType.OBJECT_DETECTION:
        run_type = "detect"
    elif run_type == InferenceType.SEGMENTATION:
        run_type = "segment"
    elif run_type == InferenceType.CLASSIFICATION:
        run_type == "classify"
    else:
        raise ValueError("invalide run_type")
    """Returns the name of the latest (most recent) file 
    of the joined path(s)"""
    fullpath = os.path.join(path,run_type, "*")
    list_of_files = glob.glob(fullpath)  # You may use iglob in Python3
    if not list_of_files:                # I prefer using the negation
        return None                      # because it behaves like a shortcut
    latest_file = max(list_of_files, key=os.path.getctime)
    _, filename = os.path.split(latest_file)
    return os.path.join(path, run_type, filename)

def get_train_infos(run_type: InferenceType):
    last_run_path = get_latest_file('runs', run_type)
    weights_path = os.path.join(last_run_path, 'weights', 'best.pt')
    results_path = os.path.join(last_run_path, 'results.csv')
    return weights_path, results_path
