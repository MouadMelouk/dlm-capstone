#from ... import function

#heatmap, prediction = function(...)

import os
import cv2
import yaml
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from detectors import DETECTOR  # Make sure this is importable in your environment

def run_inference_on_images_with_old_preprocess(
    detector_path: str,
    weights_path: str,
    image_paths: list,
    cuda: bool = True,
    manual_seed: int = None
):
    """
    Single function where we pass image paths directly (test mode) and do NO evaluation.
    This function uses the same "old" preprocessing from DeepfakeAbstractBaseDataset in test mode:
      - CV2 read (BGR -> RGB)
      - Resize to config['resolution'] using cv2.INTER_CUBIC
      - Convert to PIL
      - ToTensor
      - Normalize with config['mean'], config['std']
    No data augmentation is applied.

    Args:
        detector_path (str): Path to the detector YAML (config).
        weights_path (str): Path to the model checkpoint weights.
        image_paths (list): List of absolute image file paths to run inference on.
        cuda (bool): If True and CUDA is available, uses GPU.
        manual_seed (int): Optional manual seed for reproducibility.

    Returns:
        list: A list of predicted probabilities (floats) for each image in `image_paths`.
    """
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # 1. Load YAML config (which includes 'model_name', 'resolution', 'mean', 'std', etc.)
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Manual seed if specified
    if manual_seed is not None:
        config['manualSeed'] = manual_seed

    def init_seed(cfg):
        if cfg.get('manualSeed', None) is None:
            cfg['manualSeed'] = random.randint(1, 10000)
        random.seed(cfg['manualSeed'])
        torch.manual_seed(cfg['manualSeed'])
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg['manualSeed'])

    init_seed(config)

    # 3. Build the model from config
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    model.eval()

    # 4. Load the checkpoint weights
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print(f"Model loaded from: {weights_path}")

    # 5. Define a helper function that mimics old test-mode preprocessing
    def preprocess_image_cv2(img_path: str, cfg: dict):
        """
        - Reads an image via CV2 (BGR -> RGB).
        - Resizes to (cfg['resolution'], cfg['resolution']) using cv2.INTER_CUBIC.
        - Converts to PIL, then applies ToTensor and Normalize(mean=cfg['mean'], std=cfg['std']).
        - Returns the resulting Tensor on the correct device.
        """
        # Read via CV2
        img_path = img_path
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image at {img_path}")
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize
        target_size = cfg['resolution']
        img_rgb = cv2.resize(
            img_rgb, 
            (target_size, target_size),
            interpolation=cv2.INTER_CUBIC
        )
        # Convert to PIL
        pil_img = Image.fromarray(img_rgb)

        img_tensor_no_norm = T.ToTensor()(pil_img)  # shape (C,H,W), in [0..1]

        # ToTensor
        img_tensor = T.ToTensor()(pil_img)

        # Normalize
        mean = cfg['mean']  # e.g., [0.485, 0.456, 0.406]
        std = cfg['std']    # e.g., [0.229, 0.224, 0.225]
        img_tensor = T.Normalize(mean, std)(img_tensor)

        return img_tensor, img_tensor_no_norm

    # 6. Run inference over all images
    probabilities = []
    pred = []
    with torch.no_grad():
        for path in image_paths:
            input_tensor_norm, input_tensor_no_norm = preprocess_image_cv2(path[0], config)
            
            # Expand batch dimension and move to device
            input_tensor_norm = input_tensor_norm.unsqueeze(0).to(device)
            input_tensor_no_norm = input_tensor_no_norm.unsqueeze(0).to(device)
    
            # The data_dict now holds both versions:
            pathnew = os.path.join(*path[0].split('/')[-2:])
            data_dict = {
                'image': input_tensor_norm,      # normalized (for forward pass)
                'image_no_norm': input_tensor_no_norm,  # non-normalized (for overlay)
                'image_path': [[pathnew]]
            }
    
            output_dict = model(data_dict, inference=True)
    
            # Suppose 'prob' is your final classification probability:
            prob = output_dict['prob'].cpu().numpy()[0]
            probabilities.append(float(prob))
    
            # Hard predictions
            results = model.predict_labels(data_dict)
            pred.extend(results)

    return pred


# Example usage (comment out if you just want the function in a file):
if __name__ == "__main__":
    test_paths = [
        ["./051_DF.png"], ["./052_DF.png"],
        ["./051_real.png"], ["./052_real.png"]
    ]
    Results = run_inference_on_images_with_old_preprocess(
        detector_path="./config/detector/xception.yaml",
        weights_path="./weights/xception_best.pth",
        image_paths=test_paths,
        cuda=True,
        manual_seed=42
    )
    print("Results:", Results)


