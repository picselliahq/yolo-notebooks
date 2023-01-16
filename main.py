from utils import (
    picsell, yolo
)
import os
from picsellia.types.enums import AnnotationFileType

imdir = "data"

full_ds, train, test, val = picsell.get_picsellia_datasets(dataset_name="Retail Shelves", train_ds="First", test_ds=None)

train.download(target_path=os.path.join(imdir, 'train', 'images'))
val.download(target_path=os.path.join(imdir, 'val', 'images'))
test.download(target_path=os.path.join(imdir, 'test', 'images'))

annotation_path = full_ds.export_annotation_file(AnnotationFileType.COCO, imdir)

converter = yolo.YOLOFormatter(
    fpath=annotation_path,
    imdir=imdir,
    mode=yolo.Task.DETECTION
)

converter.convert()
yaml_fp = converter.generate_yaml(dpath=os.path.join(imdir, 'data.yaml'))