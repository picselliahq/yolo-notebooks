from picsellia import client
from picsellia.types.enums import InferenceType, Framework
from utils import (
    picsell, yolo
)
import os
import yaml


if __name__ == "__main__":
    wpath, _ = yolo.get_train_infos(run_type=InferenceType.OBJECT_DETECTION)
    client = picsell.get_picsellia_client()
    model = client.create_model(
        name=f'YOLOv8-{str(os.environ["project_name"])}',
        type=InferenceType.OBJECT_DETECTION,
        framework=Framework.PYTORCH,
        description="A simple Shelves object localization model"
    )
    model_version = model.create_version()
    with open('data/data.yaml', 'r') as f:
        r = yaml.safe_load(f)
    l = r["names"]
    labels = {str(i): v for i, v in enumerate(l)}
    model_version.update(labels=labels)
    model_version.store(name="weights", path=wpath)
    model_version.store(name="config", path='data/data.yaml')
    try:
        model_version.store(name="model-latest", path=wpath.replace('pt', 'onnx'))
    except Exception as e:
        print(e)