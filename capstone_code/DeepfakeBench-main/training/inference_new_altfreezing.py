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
import dlib


#new function for video detectors 
import os
import cv2
import yaml
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import dlib

from detectors import DETECTOR  # Make sure this is importable
# from .utils import ... (if you have any local imports)

def run_inference_on_video_with_old_preprocess(
    model_name: str,
    video_path: str,
    num_frames: int = 8,
    frame_stride: int = 3,
    cuda: bool = True,
    manual_seed: int = None,
):
    """
    Inference on a video using AltFreezing or any 3D model.
    This function:
      1) Loads the model/weights from your config,
      2) Extracts num_frames frames from the video (with stride),
      3) Performs face alignment per frame,
      4) Stacks frames into a [1, T, C, H, W] tensor,
      5) Runs inference,
      6) Returns output_dict.

    Args:
        model_name (str): e.g. "altfreezing".
        video_path (str): Path to a .mp4 or .avi video.
        num_frames (int): How many frames to sample from the video.
        frame_stride (int): Step size between frames (e.g. skip frames).
        cuda (bool): Whether to use GPU if available.
        manual_seed (int): Optional random seed for reproducibility.

    Returns:
        dict: The model’s output dictionary, e.g. containing 'prob'.
    """

    # -------------------------------------------------
    # 1) Build model from config & load weights
    # -------------------------------------------------
    model_configs = {
        "altfreezing": {
            "detector_path": "./training/config/detector/altfreezing.yaml",
            "weights_path": "./training/weights/altfreezing_best.pth",
        },
        # If you have more 3D detectors, add them here
    }

    if model_name not in model_configs:
        raise ValueError(f"Invalid model_name '{model_name}'. Only 'altfreezing' handled for video.")

    detector_path = model_configs[model_name]["detector_path"]
    weights_path = model_configs[model_name]["weights_path"]
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # Load YAML config
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds if specified
    if manual_seed is not None:
        config['manualSeed'] = manual_seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(manual_seed)

    # Instantiate the model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    model.eval()

    # Load checkpoint weights
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print(f"Model loaded from: {weights_path}")

    # -------------------------------------------------
    # 2) Read frames from the video (OpenCV)
    # -------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_rgb = []
    grabbed_frames = 0
    idx = 0

    while cap.isOpened() and grabbed_frames < num_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            # No more frames or error
            break
        if idx % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            grabbed_frames += 1
        idx += 1
    cap.release()

    if len(frames_rgb) == 0:
        raise ValueError(f"No frames read from {video_path}")

    # -------------------------------------------------
    # 3) Face alignment for each frame
    # -------------------------------------------------
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = "./training/dataset/shape_predictor_81_face_landmarks.dat"
    face_predictor = dlib.shape_predictor(predictor_path)

    resolution = config['resolution']
    mean = config['mean']
    std = config['std']

    def align_and_preprocess_frame(img_rgb, face_detector, face_predictor, resolution, mean, std):

        def enlarge_bbox(face, scale=1.4, image_shape=None):
            width = face.right() - face.left()
            height = face.bottom() - face.top()
            cx = (face.left() + face.right()) / 2
            cy = (face.top() + face.bottom()) / 2
        
            new_half_w = scale * width / 2
            new_half_h = scale * height / 2
        
            left = int(cx - new_half_w)
            right = int(cx + new_half_w)
            top = int(cy - new_half_h)
            bottom = int(cy + new_half_h)
        
            if image_shape is not None:
                h_img, w_img = image_shape[:2]
                left = max(left, 0)
                top = max(top, 0)
                right = min(right, w_img - 1)
                bottom = min(bottom, h_img - 1)
            
            return dlib.rectangle(left, top, right, bottom)

        """
        Align a single RGB frame (numpy array) using the same logic
        as preprocess_image_cv2, then convert to Torch Tensors.
        """
        # Detect faces
        faces = face_detector(img_rgb, 1)

        if len(faces) == 0:
            # No face => fallback: simple resize
            aligned_img = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        else:
            # Align largest face
            face = max(faces, key=lambda r: r.width() * r.height())
            face = enlarge_bbox(face, scale=1.4, image_shape=img_rgb.shape)

            # Keypoint extraction
            def get_keypts(image, face, predictor):
                shape = predictor(image, face)
                leye   = np.array([shape.part(36).x, shape.part(36).y]).reshape(1, 2)
                reye   = np.array([shape.part(45).x, shape.part(45).y]).reshape(1, 2)
                nose   = np.array([shape.part(30).x, shape.part(30).y]).reshape(1, 2)
                lmouth = np.array([shape.part(48).x, shape.part(48).y]).reshape(1, 2)
                rmouth = np.array([shape.part(54).x, shape.part(54).y]).reshape(1, 2)
                return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

            src_pts = get_keypts(img_rgb, face, face_predictor)

            # Canonical destination points
            dst_pts = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]
            ], dtype=np.float32)

            if resolution == 112:
                dst_pts[:, 0] += 8.0
            dst_pts[:, 0] *= (resolution / 112)
            dst_pts[:, 1] *= (resolution / 112)

            # Estimate affine transform
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            aligned_img = cv2.warpAffine(
                img_rgb, M, (resolution, resolution), flags=cv2.INTER_CUBIC
            )

        # Convert to PIL -> Torch Tensors
        pil_img = Image.fromarray(aligned_img)
        tensor_no_norm = T.ToTensor()(pil_img)
        tensor_norm = T.Normalize(mean, std)(tensor_no_norm.clone())
        return tensor_norm, tensor_no_norm

    all_frames_norm = []
    all_frames_no_norm = []

    for frame_rgb in frames_rgb:
        norm, no_norm = align_and_preprocess_frame(
            frame_rgb,
            face_detector=face_detector,
            face_predictor=face_predictor,
            resolution=resolution,
            mean=mean,
            std=std
        )
        all_frames_norm.append(norm)
        all_frames_no_norm.append(no_norm)

    # -------------------------------------------------
    # 4) Stack frames into [T, C, H, W], then unsqueeze => [1, T, C, H, W]
    # -------------------------------------------------
    clip_tensor_norm = torch.stack(all_frames_norm, dim=0)     # => [T, C, H, W]
    clip_tensor_no_norm = torch.stack(all_frames_no_norm, dim=0)  # => [T, C, H, W]

    # Unsqueeze to add batch dimension
    clip_tensor_norm = clip_tensor_norm.unsqueeze(0).to(device)     # => [1, T, C, H, W]
    clip_tensor_no_norm = clip_tensor_no_norm.unsqueeze(0).to(device)
    
    # *** PERMUTE THE TENSOR TO [B, C, T, H, W] ***
    clip_tensor_norm = clip_tensor_norm.permute(0, 2, 1, 3, 4)  # Now [1, 3, T, H, W]
    clip_tensor_no_norm = clip_tensor_no_norm.permute(0, 2, 1, 3, 4)  # Now [1, 3, T, H, W]

    data_dict = {
        'image': clip_tensor_norm,
        'image_no_norm': clip_tensor_no_norm,
    }

    # -------------------------------------------------
    # 5) Inference
    # -------------------------------------------------
    with torch.no_grad():
        output_dict = model(data_dict, inference=True, gradcam_mode="framewise")

    prob = output_dict['prob'].detach().cpu().numpy()[0]

    # -------------------------------------------------
    # 6) Extract Grad-CAM and Overlay on Frames
    # -------------------------------------------------

    def calculate_red_percentage(heatmap, red_threshold=150, non_red_threshold=100):
        """
        Compute the percentage of pixels in the heatmap that are strongly red.
        The heatmap is assumed to be in BGR format.
        """
        blue_channel, green_channel, red_channel = cv2.split(heatmap)
        red_mask = ((red_channel >= red_threshold) &
                    (blue_channel <= non_red_threshold) &
                    (green_channel <= non_red_threshold))
        total_pixels = heatmap.shape[0] * heatmap.shape[1]
        red_pixels = np.sum(red_mask)
        return (red_pixels / total_pixels) * 100.0

    # Hardcoded folder for saving Grad-CAM overlays
    gradcam_folder = "/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/gradcam/"
    os.makedirs(gradcam_folder, exist_ok=True)
        
    gradcam_tensor = output_dict.get("gradcam")  # (B, T, H, W)
    if gradcam_tensor is None:
        print("No Grad-CAM available, skipping overlay.")
    else:
        gradcam_tensor = gradcam_tensor.squeeze(0).detach().cpu().numpy()
        # Assume gradcam_tensor is obtained from compute_gradcam(), then:
        print("Grad-CAM tensor shape before aggregation:", gradcam_tensor.shape)
        # Expected: (2048, 4, 7, 7)
        
        gradcam_overlays = []
        per_frame_details = []  # List to hold details for each frame
    
        # Assume 'prob' is the clip-level probability (a single value)
        prob_value = output_dict['prob'].detach().cpu().numpy()[0].item()
        
        for t in range(gradcam_tensor.shape[0]):
            # Extract the heatmap for frame t (2D array: (H, W))
            heatmap = gradcam_tensor[t]
            # Resize heatmap to desired resolution (e.g., 224x224)
            heatmap_resized = cv2.resize(heatmap, (resolution, resolution))
            
            # --- Gamma Correction and Re-normalization ---
            gamma = 0.5  # Adjust gamma as needed
            heatmap_gamma = heatmap_resized ** gamma
            heatmap_norm = (heatmap_gamma - heatmap_gamma.min()) / (heatmap_gamma.max() - heatmap_gamma.min() + 1e-8)
            heatmap_uint8 = np.uint8(255 * heatmap_norm)
            # -----------------------------------------------
            
            # Apply the colormap to get a colored heatmap (BGR)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Compute red percentage for this frame's heatmap
            red_pct = calculate_red_percentage(heatmap_color)
            
            # Convert the corresponding original unnormalized frame to BGR for overlay
            frame_rgb = all_frames_no_norm[t].permute(1, 2, 0).cpu().numpy()  # (H, W, C) in [0,1]
            frame_uint8 = np.uint8(255 * frame_rgb)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            
            # Blend the heatmap overlay with the original frame
            overlay = cv2.addWeighted(frame_bgr, 0.6, heatmap_color, 0.4, 0)
            gradcam_overlays.append(overlay)
            overlay_path = os.path.join(gradcam_folder, f"gradcam_frame_{t:03d}.png")
            
            # Append per-frame details: frame index, probability, red percentage.
            per_frame_details.append((t, overlay_path, red_pct))
        
        # Optionally, return per_frame_details if needed.
        # For example:
        return prob, per_frame_details

        

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
        predictor_path = './training/dataset/shape_predictor_81_face_landmarks.dat'
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
            
            # --- Wrap the image to get an aligned, cropped face ---
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images or video with a specified model.")
    parser.add_argument("model_name", type=str, choices=["ucf", "xception", "spsl", "altfreezing"], 
                        help="Name of the model to use for inference. Choices: 'ucf', 'xception', 'spsl', and 'altfreezing'.")
    parser.add_argument("image_paths", nargs="+", type=str, 
                        help="Paths to the images for inference. Provide one or more image paths.")
    parser.add_argument("--video", action="store_true", help="If set, treat 'path' as a video file.") #if arg provided, treat path as video
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=3)

    args = parser.parse_args()

   
    if args.video:
        # Call the VIDEO function
        video_path = args.image_paths[0]
        result, frame_details = run_inference_on_video_with_old_preprocess(
            model_name=args.model_name,
            video_path=video_path,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            cuda=True,
            manual_seed=42,
        )
        print("Video Inference:", result)
        threshold= 0.5  # Adjust threshold based on model calibration
        decision = "detected" if result > threshold else "not detected"
        explanation = f"Frequency model has **{decision}** signature forgery spacial and temporal with probability {result}."
        print(explanation)
        print(frame_details)

    else:
        
        test_paths = [[path] for path in args.image_paths]
        Results = run_inference_on_images_with_old_preprocess(
            model_name=args.model_name,
            image_paths=test_paths,
            cuda=True,
            manual_seed=42,
        )
        print(Results)
