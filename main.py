from utils import (
    picsell, yolo
)
import os
from picsellia.types.enums import AnnotationFileType
from ultralytics import YOLO
from ultralytics.yolo.v8.detect import DetectionTrainer

if __name__ == "__main__":

    imdir = "data"

    full_ds, train, test, val = picsell.get_picsellia_datasets(dataset_name="Retail Shelves", train_ds="First", test_ds=None)

    train.download(target_path=os.path.join(imdir, 'train', 'images'), max_workers=10)
    val.download(target_path=os.path.join(imdir, 'val', 'images'), max_workers=10)
    test.download(target_path=os.path.join(imdir, 'test', 'images'), max_workers=10)

    annotation_path = full_ds.export_annotation_file(AnnotationFileType.COCO, imdir)

    converter = yolo.YOLOFormatter(
        fpath=annotation_path,
        imdir=imdir,
        mode=yolo.Task.DETECTION
    )

    converter.convert()
    yaml_fp = converter.generate_yaml(dpath=os.path.join(imdir, 'data.yaml'))

    yaml_fp = "data/data.yaml"
    trainer = DetectionTrainer(overrides={"data": yaml_fp, "model": "yolov8n.pt", "epochs": 5, "pretrained": True, "device": "cpu"})
    model = trainer.train()
    results = model.val()
    success = model.export(format="onnx")