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
import cv2
import dlib
import numpy as np
from PIL import Image
import torchvision.transforms as T

def run_inference_on_images_with_old_preprocess(
    model_name: str,
    image_paths: list,
    cuda: bool = True,
    manual_seed: int = None,
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
        model_name (str): One of "spsl", "xception", or "ucf" to select the detector and weight paths.
        image_paths (list): List of absolute image file paths to run inference on.
        cuda (bool): If True and CUDA is available, uses GPU.
        manual_seed (int): Optional manual seed for reproducibility.

    Returns:
        list: A list of predicted probabilities (floats) for each image in `image_paths`.
    """

    print(f"Called with model: {model_name.upper()}")
    
    model_configs = {
        "spsl": {
            "detector_path": "./config/detector/spsl.yaml",
            "weights_path": "./weights/spsl_best.pth",
        },
        "xception": {
            "detector_path": "./config/detector/xception.yaml",
            "weights_path": "./weights/xception_best.pth",
        },
        "ucf": {
            "detector_path": "./config/detector/ucf.yaml",
            "weights_path": "./weights/ucf_best.pth",
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"Invalid model_name '{model_name}'. Choose from 'spsl', 'xception', or 'ucf'.")

    detector_path = model_configs[model_name]["detector_path"]
    weights_path = model_configs[model_name]["weights_path"]

    # Call the actual inference function with resolved paths
    return run_inference_on_images_with_old_preprocess_core(
        detector_path=detector_path,
        weights_path=weights_path,
        image_paths=image_paths,
        cuda=cuda,
        manual_seed=manual_seed,
    )


def run_inference_on_images_with_old_preprocess_core(
    detector_path: str,
    weights_path: str,
    image_paths: list,
    cuda: bool = True,
    manual_seed: int = None,
):
    """
    Core function that runs inference using the provided detector and weight paths.
    Separated to keep logic clean when selecting model configurations.
    """
    running_inference = False if "spsl" in weights_path else True
    
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
        Reads an image via CV2, performs face detection & alignment (with cropping),
        and then produces both a normalized and non-normalized tensor.
        Additionally, returns the crop coordinates (in the original image) used during alignment.
        """
        # --- Load and convert image ---
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image at {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # --- Face detection and alignment setup ---
        face_detector = dlib.get_frontal_face_detector()
        predictor_path = './dataset/shape_predictor_81_face_landmarks.dat'
        face_predictor = dlib.shape_predictor(predictor_path)
        
        # --- Detect face ---
        faces = face_detector(img_rgb, 1)
        if len(faces) == 0:
            # No face detected; fallback to simple resize.
            aligned_img = cv2.resize(
                img_rgb, (cfg['resolution'], cfg['resolution']),
                interpolation=cv2.INTER_CUBIC
            )
            crop_coords = None
        else:
            # Select the largest face.
            face = max(faces, key=lambda r: r.width() * r.height())
            
            # --- Extract five keypoints ---
            def get_keypts(image, face, predictor):
                shape = predictor(image, face)
                # Note: dlib's 68-landmark indexing is 0-indexed.
                leye   = np.array([shape.part(36).x, shape.part(36).y]).reshape(1, 2)
                reye   = np.array([shape.part(45).x, shape.part(45).y]).reshape(1, 2)
                nose   = np.array([shape.part(30).x, shape.part(30).y]).reshape(1, 2)
                lmouth = np.array([shape.part(48).x, shape.part(48).y]).reshape(1, 2)
                rmouth = np.array([shape.part(54).x, shape.part(54).y]).reshape(1, 2)
                return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
            
            src_pts = get_keypts(img_rgb, face, face_predictor)
            
            # --- Define canonical destination points ---
            target_size = cfg['resolution']  # e.g., 256
            dst_pts = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]
            ], dtype=np.float32)
            if target_size == 112:
                dst_pts[:, 0] += 8.0
            # Scale from the canonical 112x112 to target size.
            dst_pts[:, 0] = dst_pts[:, 0] * target_size / 112
            dst_pts[:, 1] = dst_pts[:, 1] * target_size / 112
            
            # --- Compute similarity transform ---
            # Using cv2.estimateAffinePartial2D to get a 2x3 matrix.
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            
            # --- Warp the image to get an aligned, cropped face ---
            aligned_img = cv2.warpAffine(
                img_rgb, M, (target_size, target_size), flags=cv2.INTER_CUBIC
            )
            
            # --- Compute crop coordinates in the original image ---
            # Destination corners in the aligned image.
            dst_corners = np.array([
                [0, 0],
                [target_size - 1, 0],
                [target_size - 1, target_size - 1],
                [0, target_size - 1]
            ], dtype=np.float32)
            # Convert to homogeneous coordinates.
            dst_corners_h = np.hstack([dst_corners, np.ones((4, 1), dtype=np.float32)])
            # Extend M to 3x3.
            M_full = np.vstack([M, [0, 0, 1]])
            # Invert the transformation.
            M_inv = np.linalg.inv(M_full)
            # Map destination corners back to original image coordinates.
            src_corners = (M_inv @ dst_corners_h.T).T[:, :2]
            crop_coords = src_corners.astype(np.int32)
        
        # --- Final processing: convert aligned image to tensors ---
        pil_img = Image.fromarray(aligned_img)
        img_tensor_no_norm = T.ToTensor()(pil_img)
        img_tensor = T.Normalize(cfg['mean'], cfg['std'])(img_tensor_no_norm.clone())
        
        return img_tensor, img_tensor_no_norm, crop_coords


    # 6. Run inference over all images
    probabilities = []
    pred = []
    with torch.no_grad():
        for path in image_paths:
            # Unpack three values instead of two.
            input_tensor_norm, input_tensor_no_norm, crop_coords = preprocess_image_cv2(path[0], config)
            
            # Expand batch dimension and move to device.
            input_tensor_norm = input_tensor_norm.unsqueeze(0).to(device)
            input_tensor_no_norm = input_tensor_no_norm.unsqueeze(0).to(device)
    
            pathnew = os.path.join(*path[0].split('/')[-2:])
            data_dict = {
                'image': input_tensor_norm,             # normalized (for forward pass)
                'image_no_norm': input_tensor_no_norm,     # non-normalized (for overlay)
                'crop_coords': crop_coords,               # now available for further use
                'image_path': [[pathnew]]
            }
    
            output_dict = model(data_dict, inference=running_inference)

            # Suppose 'prob' is your final classification probability:
            prob = output_dict['prob'].cpu().numpy()[0]
            probabilities.append(float(prob))
    
            # Hard predictions
            results = model.predict_labels(data_dict)
            pred.extend(results)

    return pred


import argparse

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images with a specified model.")
    parser.add_argument("model_name", type=str, choices=["ucf", "xception", "spsl"], 
                        help="Name of the model to use for inference. Choices: 'ucf', 'xception', 'spsl'.")
    parser.add_argument("image_paths", nargs="+", type=str, 
                        help="Paths to the images for inference. Provide one or more image paths.")

    args = parser.parse_args()

    # Wrap each image path in a list, maintaining the original nested structure
    test_paths = [[path] for path in args.image_paths]

    Results = run_inference_on_images_with_old_preprocess(
        model_name=args.model_name,
        image_paths=test_paths,
        cuda=True,
        manual_seed=42,
    )
    print(Results)
