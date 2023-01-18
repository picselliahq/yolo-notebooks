import picsellia 
import ignite
import torch
import torchvision 
import os
from datetime import datetime
from typing import List, Tuple
from PIL import Image
from picsellia import Client
from picsellia.sdk.project import Project as PicselliaProject
from picsellia.sdk.experiment import Experiment as PicselliaExperiment
from picsellia.sdk.dataset_version import DatasetVersion as PicselliaDatasetVersion
from picsellia.sdk.deployment import Deployment as DeployedPicselliaModel
from picsellia.sdk.asset import MultiAsset
import tqdm 
import platform 
import torch
import random


def get_picsellia_client(organization_name: str = None) -> Client:
    api_token = os.environ.get("api_token")
    if api_token is None:
        api_token = input("Please enter your TOKEN here :")
    organization_id = os.environ.get("organization_id")
    if organization_id is None:
        organization_id = input("Please enter your ORGANIZATorganization_idON_ID here :")
    
    return picsellia.Client(api_token=api_token, organization_id=organization_id)

def split_ds_in_train_test_split(full_ds: PicselliaDatasetVersion, random_seed = 42):
    train_assets, eval_assets, count_train, count_eval, labels = full_ds.train_test_split(random_seed=random_seed)
    eval_list = eval_assets.items.copy()
    random.shuffle(eval_list)
    nb_val = len(eval_assets.items)//2
    val_data = eval_list[nb_val:]
    test_data = eval_list[:nb_val]
    val_assets = MultiAsset(full_ds.connexion, full_ds.id, val_data)
    test_assets = MultiAsset(full_ds.connexion, full_ds.id, test_data)
    return (train_assets, test_assets, val_assets)


def get_picsellia_project() -> PicselliaProject:
    project_name = os.environ.get("project_name")
    if project_name is None:
        project_name = input("Please enter your project_name here :")
    return get_picsellia_client().get_project(project_name=project_name)


def get_picsellia_experiment() -> PicselliaExperiment:
    experiment_name = os.environ.get("experiment_name")
    if experiment_name is None:
        experiment_name = input("Please enter your experiment_name here :")
    return get_picsellia_project().get_experiment(name=experiment_name)


def get_picsellia_datasets(
    dataset_name: str = None,
    train_ds:  str = "train", test_ds: str = "test",
    val_ds: str = "val") -> List[PicselliaDatasetVersion]:
        """ 
        Assuming that you have 3 versions of a Dataset "train", "test", "split"
        """
        with get_picsellia_client() as client:
            dataset = client.get_dataset(dataset_name)
            train_ds = dataset.get_version(train_ds)
            if test_ds is None or val_ds is None:
                return (train_ds,) + split_ds_in_train_test_split(train_ds, random_seed=42) # TODO -> make it a variable
            test_ds = dataset.get_version(test_ds)
            val_ds = dataset.get_version(val_ds)
        return (None,) + (train_ds, test_ds, val_ds)
    
    

def checkout_project(client: Client, project_name: str = None) -> PicselliaProject:
    if project_name is None:
        project_name = input("Please enter your project_name here :")
    return client.get_project(project_name)


def generate_new_experiment(project: PicselliaProject = None, visualize_test: bool = False) -> PicselliaExperiment:
    """ 
    This utility function assume that you have at least 3 datasets in your project called
    ('train', 'test', 'valid'),
    This function will: 
       - Create a new experiment 
       - Attach the 3 datasets to this experiment
       - if verbose=True:
            - Will create a fork of the test set to display the predictions as annotations.
    
    Returns:
        PicselliaExperiment Object
    """
    datasets: List[PicselliaDatasetVersion] = project.list_dataset_versions()
    print(f"{len(datasets)} included in your project:")
    for dataset in datasets:
        print(f"{dataset.name}/{dataset.version}")
        if dataset.version == "train":
            train_dataset = dataset 
        elif dataset.version == "test":
            test_dataset = dataset 
        elif dataset.version == "valid":
            valid_dataset = dataset 
    experiment: PicselliaExperiment = project.create_experiment(
        name=f"v{len(project.list_experiments())} - training"
    )
    experiment.attach_dataset(name="train", dataset_version=train_dataset)
    experiment.attach_dataset(name="test", dataset_version=test_dataset)
    experiment.attach_dataset(name="valid", dataset_version=valid_dataset)

    if visualize_test: # We will generate a fork of the validation dataset to store the predictions and vizualize them
        test_dataset_prediction, job = valid_dataset.fork(version=f"test - prediction - {experiment.name}", assets=valid_dataset.list_assets(), type=valid_dataset.type)
        job.wait_for_done()
        experiment.attach_dataset(name="test-prediction", dataset_version=test_dataset_prediction)
    return experiment

def get_train_test_valid_datasets(experiment: PicselliaExperiment) -> Tuple[PicselliaDatasetVersion]:
    train: PicselliaDatasetVersion = experiment.get_dataset('train')
    test : PicselliaDatasetVersion = experiment.get_dataset('test')
    valid: PicselliaDatasetVersion = experiment.get_dataset('valid')
    return (train, test, valid)


    
