"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
parser.add_argument("--lmdb", type=bool, default=False)
parser.add_argument("--pairstride", type=int, default=1, help="Pair stride to use for testing")
parser.add_argument("--srcstride", type=int, default=16, help="Source stride to use for testing")
parser.add_argument("--use_flows", action="store_true", help="Use flows if this flag is set")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test'
        )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=1, #hardcoded batch size 1
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset_GRADCAM(model, data_loader, output_dir="gradcam_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    prediction_lists = []
    feature_lists = []
    label_lists = []

    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Get data
        data, label, mask, landmark, image_path = (
            data_dict['image'],
            data_dict['label'],
            data_dict['mask'],
            data_dict['landmark'],
            data_dict['image_path'],
        )

        # Ensure image_path is a string
        print(f"image_path: {image_path}")
        image_path = image_path[0] if isinstance(image_path, list) else image_path
        print(f"image_path after manipulation: {image_path}")
        
        # Convert label for binary classification
        label = torch.where(data_dict['label'] != 0, 1, 0)

        # Move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # Model forward without considering gradient computation
        predictions = inference(model, data_dict)

        # Generate and save GradCAM visualization
        model.generate_gradcam(data, target_class=1, image_path=image_path)
        
        # Append results
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
        torch.cuda.empty_cache()
    
    return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)


def modify_path(original_image_path: str, model_type: str, use_flows: bool) -> str:
    if use_flows:
        if "/flows/" in original_image_path:
            new_path = original_image_path.replace("/flows/", f"/flow_embeddings/{model_type}/")
        else:
            raise ValueError("Not implemented, flows path not found")
    else:
        if "/frames/" in original_image_path:
            new_path = original_image_path.replace("/frames/", f"/frame_embeddings/{model_type}/")
        else:
            raise ValueError("Not implemented, frames path not found")
    
    # Replace .png with .npy
    new_path = new_path.replace(".png", ".npy")
    
    # Ensure parent directory exists
    parent_dir = os.path.dirname(new_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    return new_path


def test_one_dataset_EMBEDDINGS(model, data_loader, model_type, use_flows):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Get data
        data, label, _, image_path = (
            data_dict['image'],
            data_dict['label'],
            data_dict['mask'],
            data_dict['name'],
        )
        
        # Ensure image_path is a string
        #print(f"image_path   : {image_path}")
        #print(f"image_path[0]: {image_path[0]}")
        #print(f"image_path[0][0]: {image_path[0][0]}")
        
        # Convert label for binary classification
        label = torch.where(label != 0, 1, 0)
    
        # Move data to GPU
        data = data.to(device)
        label = label.to(device)

        # Prepare data_dict for model
        data_dict = {'image': data, 'label': label}

        # Model forward without considering gradient computation
        predictions = inference(model, data_dict)

        # Generate and save embedding
        save_path = modify_path(image_path[0][0], model_type, use_flows) # batch size is always 1, do not change this logic
        model.generate_embedding(data, save_path)
        
    return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)

def test_epoch(model, test_data_loaders, model_type, use_flows):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset, save embeddings
        predictions_nps, label_nps,feat_nps = test_one_dataset_EMBEDDINGS(model, test_data_loaders[key], model_type, use_flows)
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    if args.pairstride:
        config['pairstride'] = args.pairstride
    if args.srcstride:
        config['srcstride'] = args.srcstride
    config['use_flows'] = args.use_flows  # Boolean, True if flag is passed

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    print(config['model_name'])
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders, config['model_name'], config['use_flows'])
    print('===> Test Done!')

if __name__ == '__main__':
    main()
