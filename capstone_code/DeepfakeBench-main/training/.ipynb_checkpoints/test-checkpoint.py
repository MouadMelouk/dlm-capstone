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
import torchvision.transforms as T


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)
parser.add_argument('--test_image', type=str, help="Path to a single image for testing (optional)")
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
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
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


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

def test_single_image_GRADCAM(model, image_path, output_folder="/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/gradcams"):

    # Load and preprocess image
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to match model input
        T.ToTensor(),
    ])

    image = pil_image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Create a dummy data_dict
    data_dict = {
        "image": image_tensor,
        "label": torch.tensor([0]).to(device),  # Default label (not used for inference)
        "mask": None,
        "landmark": None,
        "image_path": [image_path],
    }

    # Model inference
    predictions = inference(model, data_dict)

    if 'prob' in predictions:
        score = predictions['prob'].cpu().detach().numpy()[0]  # Standard case
    elif 'cls' in predictions:
        score = torch.softmax(predictions['cls'], dim=1)[:, 1].cpu().detach().numpy()[0]  # Convert logits to prob
    else:
        raise KeyError("Neither 'prob' nor 'cls' found in model output.")


    # Generate Grad-CAM overlay
    gradcam_image = model.generate_gradcam(image_tensor, target_class=1)

    # Hardcoded Save Path
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    image_name = os.path.basename(image_path)  # Extract original image name
    save_name = f"{os.path.splitext(image_name)[0]}_GradCAM.png"  # Append "_GradCAM"
    save_path = os.path.join(output_folder, save_name)  # Hardcoded output directory

    # Save the Grad-CAM Image
    cv2.imwrite(save_path, gradcam_image)  # Save heatmap
    print(f"✅ Grad-CAM saved at: {save_path}")

    # Convert score to explanation
    threshold = 0.5  # Adjust threshold based on model calibration
    decision = "detected" if score > threshold else "not detected"
    explanation = f"Frequency model **{decision}** signature forgery frequencies with confidence of {score:.4f}."

    return gradcam_image, explanation  # Return overlay image, explanation, and save path

def run_gradcam_for_single_image(detector_path, weights_path, image_path, cuda: bool = True, manual_seed: int = None):
    """
    Loads the model, prepares the config, and runs Grad-CAM on a single image.

    Args:
        detector_path (str): Path to the detector YAML config.
        weights_path (str): Path to the model weights.
        image_path (str): Path to the single image to test.
        output_folder (str): Directory where the Grad-CAM heatmap should be saved.

    Returns:
        tuple: (Grad-CAM image, explanation string)
    """
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model configuration
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)

    #manual seed 
    # 2. Manual seed if specified
    if manual_seed is not None:
        config['manualSeed'] = manual_seed

    def init_seed(config):
        if config['manualSeed'] is None:
            config['manualSeed'] = random.randint(1, 10000)
        random.seed(config['manualSeed'])
        torch.manual_seed(config['manualSeed'])
        if config['cuda']:
            torch.cuda.manual_seed_all(config['manualSeed'])

    # Initialize seed for reproducibility
    init_seed(config)

    # Load the model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    model.eval()

    # Load pre-trained weights
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)  # Allow missing keys for robustness
    print(f" Model loaded from: {weights_path}")

    # Lists to store results
    gradcam_paths = []
    explanations = []

    # Process each image
    with torch.no_grad():
        for img_path in image_paths:
            # Ensure img_path is a string if passed as a list
            img_path = img_path[0] if isinstance(img_path, list) else img_path

            # Run Grad-CAM for the current image
            gradcam_image, explanation = test_single_image_GRADCAM(
                model=model,
                image_path=img_path,
            )

            # Store results
            gradcam_paths.append(gradcam_image)
            explanations.append(explanation)

    return gradcam_paths, explanations

def test_single_image_GRADCAM(model, image_path, output_folder="/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/gradcams"):

    # Load and preprocess image
    image_tensor = preprocess_image_cv2(image_path, config)

    data_dict = {
    "image": image_tensor,  # Now using colleague's function output
    "label": torch.tensor([0]).to(device),  # Default label (not used for inference)
    "mask": None,
    "landmark": None,
    "image_path": [image_path],
}

    # Model inference
    predictions = inference(model, data_dict)

    if 'prob' in predictions:
        score = predictions['prob'].cpu().detach().numpy()[0]  # Standard case
    elif 'cls' in predictions:
        score = torch.softmax(predictions['cls'], dim=1)[:, 1].cpu().detach().numpy()[0]  # Convert logits to prob
    else:
        raise KeyError("Neither 'prob' nor 'cls' found in model output.")


    # Generate Grad-CAM overlay
    gradcam_image = model.generate_gradcam(image_tensor, target_class=1)

    # Hardcoded Save Path
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    image_name = os.path.basename(image_path)  # Extract original image name
    save_name = f"{os.path.splitext(image_name)[0]}_GradCAM.png"  # Append "_GradCAM"
    save_path = os.path.join(output_folder, save_name)  # Hardcoded output directory

    # Save the Grad-CAM Image
    cv2.imwrite(save_path, gradcam_image)  # Save heatmap
    print(f"✅ Grad-CAM saved at: {save_path}")

    # Convert score to explanation
    threshold = 0.5  # Adjust threshold based on model calibration
    decision = "detected" if score > threshold else "not detected"
    explanation = f"Frequency model **{decision}** signature forgery frequencies with confidence of {score:.4f}."

    return gradcam_image, explanation  # Return overlay image, explanation, and save path

    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])
        
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
    # Load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)

    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']

    # Overwrite settings from command-line arguments
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    else:
        print("Error: No model weights provided. Use --weights_path")
        return

    # Init seed
    init_seed(config)

    # Set cudnn benchmark if needed
    if config.get('cudnn', False):
        cudnn.benchmark = True

    # Prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    # Load model weights
    try:
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)  # Allow missing keys for robustness
        print("===> Load checkpoint done!")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    # Set model to eval mode
    model.eval()

    # === Test Single Image ===
    if hasattr(args, "test_image") and args.test_image:
        gradcam_image, explanation = test_single_image_GRADCAM(model, args.test_image)

        print("\nFinal Output:")
        print(explanation)
        return  # Exit after testing one image

    # === Test Full Dataset ===
    if args.test_dataset:
        print("===> Running full dataset test.")
        test_data_loaders = prepare_testing_data(config)
        best_metric = test_epoch(model, test_data_loaders)
        print("===> Test Done!")
    else:
        print("Error: You must specify either --test_image or --test_dataset.")


if __name__ == '__main__':
    main()
