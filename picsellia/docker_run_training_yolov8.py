import os
import shutil
import logging
import pandas as pd
from ultralytics import YOLO
from picsellia.types.enums import (
    AnnotationFileType, LogType
)
from utils import (
    picsell, yolo
)

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True"

logging.getLogger('picsellia').setLevel(logging.INFO)

if __name__ == "__main__":
    datasets_dir = 'datasets'
    imdir = "data"
    if not os.path.isdir(datasets_dir):
        os.mkdir(datasets_dir)
    if not os.path.isdir(imdir):
        os.mkdir(imdir)

    if not os.path.isdir("runs"):
        os.mkdir("runs")

    experiment = picsell.get_experiment()
    train_ds, test_ds, val_ds = picsell.get_train_test_valid_datasets(experiment=experiment)
    
    experiment.start_logging_chapter('Dowloading files')
    
    trained_weights = experiment.get_artifact("model-latest")
    trained_weights.download()
    weights_path = trained_weights.filename
        
    parameters = experiment.get_log('parameters').data

    epochs = parameters.get("epochs", 30) # try to find the number of epochs in parameter, otherwise set to 30
    batch_size = parameters.get("batch_size", 8)
    
    train_ds.download(target_path=os.path.join(imdir, 'train', 'images'), max_workers=10)
    val_ds.download(target_path=os.path.join(imdir, 'val', 'images'), max_workers=10)
    test_ds.download(target_path=os.path.join(imdir, 'test', 'images'), max_workers=10)

    formatter = yolo.YOLOFormatter(
        fpath=train_ds.export_annotation_file(AnnotationFileType.COCO, imdir),
        imdir=imdir,
        mode=train_ds.type,
        steps="train"
    )
    formatter.convert()

    yolo.YOLOFormatter(
        fpath=val_ds.export_annotation_file(AnnotationFileType.COCO, imdir),
        imdir=imdir,
        mode=train_ds.type,
        steps="val"
    ).convert()
    yolo.YOLOFormatter(
        fpath=test_ds.export_annotation_file(AnnotationFileType.COCO, imdir),
        imdir=imdir,
        mode=train_ds.type,
        steps="test"
    ).convert()

    yaml_fp = formatter.generate_yaml(dpath=os.path.join(imdir, 'data.yaml'))
    
    shutil.move(imdir, datasets_dir)
    if not os.path.isdir(imdir):
        os.mkdir(imdir)
    shutil.move('datasets/data/data.yaml', 'data/data.yaml')

    experiment.start_logging_chapter('Init Model')

    model = YOLO(weights_path)
    
    experiment.start_logging_chapter('Training')
    
    results = model.train(data=yaml_fp, epochs=epochs, batch=batch_size, device='0')
    
    wpath, rpath = yolo.get_train_infos(train_ds.type)
    try:
        success = model.export(format="onnx")
        experiment.store('weights', path=wpath)
        experiment.store('model-latest', path=wpath.replace('pt', 'onnx'))

        res = pd.read_csv(rpath)
        for col in res.columns:
            try:
                experiment.log(name=str(col), data=list(res[col]), type=LogType.LINE)
            except Exception as e:
                print(e)
    except Exception as e:
        print("failed to export ONNX model")
        experiment.store('weights', path=wpath)
        res = pd.read_csv(rpath)
        for col in res.columns:
            try:
                experiment.log(name=str(col), data=list(res[col]), type=LogType.LINE)
            except Exception as e:
                print(e)
    experiment.export_as_model(name=f"YOLOv8-{experiment.name}")



