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
from preprocessing.preprocess import extract_aligned_face_dlib
import argparse
import json

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

    #print(f"Called with model: {model_name.upper()}")
    
    model_configs = {
        "spsl": {
            "detector_path": "/scratch/rz2288/DeepfakeBench/training/config/detector/spsl.yaml",
            "weights_path": "/scratch/rz2288/DeepfakeBench/training/weights/spsl_best.pth",
        },
        "xception": {
            "detector_path": "/scratch/rz2288/DeepfakeBench/training/config/detector/xception.yaml",
            "weights_path": "/scratch/rz2288/DeepfakeBench/training/weights/xception_best.pth",
        },
        "ucf": {
            "detector_path": "/scratch/rz2288/DeepfakeBench/training/config/detector/ucf.yaml",
            "weights_path": "/scratch/rz2288/DeepfakeBench/training/weights/ucf_best.pth",
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
    #print(f"Model loaded from: {weights_path}")

    # 5. Define a helper function that mimics old test-mode preprocessing

    def preprocess_image_cv2(img_path: str, cfg: dict):
        """
        Reads an image via CV2, performs face detection & alignment (with cropping),
        and then produces both a normalized and non-normalized tensor.
        Additionally, returns the crop coordinates (in the original image) used during alignment.
        """
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
        predictor_path = './preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'
        face_predictor = dlib.shape_predictor(predictor_path)
        
        # --- Detect face and align using the provided logic ---
        cropped_face, landmarks, mask_face = extract_aligned_face_dlib(
            face_detector, face_predictor, img_rgb, res=cfg['resolution'], mask=None
        )
        
        if cropped_face is None:
            # No face detected; fallback to simple resize.
            aligned_img = cv2.resize(
                img_rgb, (cfg['resolution'], cfg['resolution']),
                interpolation=cv2.INTER_CUBIC
            )
            crop_coords = None
        else:
            # Use the aligned and cropped face
            aligned_img = cropped_face
            # Compute crop coordinates in the original image
            # (This part is not implemented in the provided code, so we leave it as None)
            crop_coords = None
        
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

def extract_random_frames(video_path, output_folder, num_frames=5):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return []

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure the number of frames requested is not greater than the total frames
    if num_frames > total_frames:
        num_frames = total_frames
        print(f"Warning: Only {total_frames} frames available in {video_path}. Extracting all.")

    # Randomly select frame indices
    frame_indices = sorted(random.sample(range(total_frames), num_frames))

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_paths = []

    for idx in frame_indices:
        # Set the video to the selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {idx} from {video_path}.")
            continue

        # Save the frame
        frame_filename = os.path.join(output_folder, f"frame_{idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_paths.append(frame_filename)

    cap.release()
    return frame_paths

def process_videos(video_list, output_folder, num_frames=5):
    all_frame_paths = []

    for video_path in video_list:
        #print(f"Processing {video_path}...")
        frame_paths = extract_random_frames(video_path, output_folder, num_frames)
        all_frame_paths.extend(frame_paths)

    return all_frame_paths



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images with a specified model.")
    parser.add_argument("model_name", type=str, choices=["ucf", "xception", "spsl"], 
                        help="Name of the model to use for inference. Choices: 'ucf', 'xception', 'spsl'.")
    parser.add_argument("image_paths", nargs="+", type=str, 
                        help="Paths to the images for inference. Provide one or more image paths.")

    args = parser.parse_args()

    # Wrap each image path in a list, maintaining the original nested structure
    test_paths = [[path] for path in args.image_paths]

    

    # Path to your JSON file
    #json_file_path = "/scratch/rz2288/DeepfakeBench/preprocessing/dataset_json/Celeb-DF-v1.json"
    
    #with open(json_file_path, "r") as f:
        data = json.load(f)
    
    # Extract filenames from JSON
    #FF_Fsh_test_paths = list(data["Celeb-DF-v1"]["CelebDFv1_fake"]["test"].keys())
    
    # Construct absolute file paths
    #all_test_paths = ["/scratch/rz2288/DeepfakeBench/datasets/rgb/Celeb-DF-v1/Celeb-synthesis/" + filename + ".mp4" for filename in FF_Fsh_test_paths][:5]

    #output_folder = "output_frames_0308_1"
    #num_frames = 5  # Number of random frames to extract from each video

    #frame_paths = process_videos(all_test_paths, output_folder, num_frames)


    # Wrap each image path in a list, maintaining the original nested structure
    #test_paths = [["/scratch/rz2288/DeepfakeBench/"+path] for path in frame_paths]

    Results = run_inference_on_images_with_old_preprocess(
        model_name=args.model_name,
        image_paths=test_paths,
        cuda=True,
        manual_seed=42,
    )
    print(Results)