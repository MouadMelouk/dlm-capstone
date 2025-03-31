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
import json 
from pathlib import Path
from sklearn.metrics import roc_auc_score


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
from imutils import face_utils
from skimage import transform as trans
from typing import Tuple, Union
from preprocessing.preprocess import get_keypts

from detectors import DETECTOR  # Make sure this is importable
# from .utils import ... (if you have any local imports)

def extract_aligned_face_dlib(
    face_detector, predictor, image: np.ndarray, res: int = 256, mask=None
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Detects, aligns, and crops the largest face in an image using 5-point landmarks.
    """
    def img_align_crop(img, landmark=None, outsize=(256, 256), scale=1.3, mask=None):
        """
        Align and crop the face using 5 facial landmarks.
        """
        # Target positions of 5 landmarks in a standard aligned face
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if outsize[1] == 112:
            dst[:, 0] += 8.0  # Shift X-coordinates for standard alignment

        # Scale transformation based on output size
        dst[:, 0] = dst[:, 0] * outsize[0] / 112
        dst[:, 1] = dst[:, 1] * outsize[1] / 112

        # Apply scale margin
        margin_rate = scale - 1
        x_margin = outsize[0] * margin_rate / 2.0
        y_margin = outsize[1] * margin_rate / 2.0
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # Normalize back after adding margins
        dst[:, 0] *= outsize[0] / (outsize[0] + 2 * x_margin)
        dst[:, 1] *= outsize[1] / (outsize[1] + 2 * y_margin)

        # Compute affine transform from detected landmarks → target positions
        src = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]  # 2x3 affine transformation matrix

        # Apply affine transformation
        aligned_face = cv2.warpAffine(img, M, (outsize[1], outsize[0]))

        if mask is not None:
            mask = cv2.warpAffine(mask, M, (outsize[1], outsize[0]))
            return aligned_face, mask
        return aligned_face, None

    # Convert image to RGB (dlib expects RGB)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_detector(rgb_img, 1)
    if len(faces) == 0:
        return None, None, None  # No face detected

    # Choose the **largest face**
    face = max(faces, key=lambda rect: rect.width() * rect.height())

    # Extract **5-point landmarks**
    landmarks = get_keypts(rgb_img, face, predictor, face_detector)

    # Align and crop the face
    aligned_face, mask_face = img_align_crop(rgb_img, landmarks, outsize=(res, res), mask=mask)
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)  # Convert back to BGR

    # Extract additional landmarks from the aligned face
    face_align = face_detector(aligned_face, 1)
    if len(face_align) == 0:
        return None, None, None  # Alignment failed

    aligned_landmark = predictor(aligned_face, face_align[0])
    aligned_landmark = face_utils.shape_to_np(aligned_landmark)

    return aligned_face, aligned_landmark, mask_face


'''def enlarge_bbox(face, scale=1.7, image_shape=None):
    """
    Expands the detected face bounding box by a scaling factor.
    Ensures the new bounding box does not exceed image boundaries.
    """
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

    return dlib.rectangle(left, top, right, bottom)''' 


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
            cv2.imwrite(overlay_path, overlay)
            per_frame_details.append((t, overlay_path, red_pct))
        
        # Optionally, return per_frame_details if needed.
        # For example:
        return prob, per_frame_details

def run_inference_multiple_clips(
    model_name: str,
    video_path: str,
    num_frames: int = 8,
    num_clips: int = 4,  # Extract exactly 4 clips (4x8 = 32 frames total)
    gradcam_aggregation: str = "mean",  # Options: "mean", "max", "last_frame"
    cuda: bool = True,
    manual_seed: int = None,
):

    # -------------------------------------------------
    # 1) Load Model & Config
    # -------------------------------------------------
    model_configs = {
        "altfreezing": {
            "detector_path": "./training/config/detector/altfreezing.yaml",
            "weights_path": "./training/weights/altfreezing_best.pth",
        },
    } 

    if model_name not in model_configs:
        raise ValueError(f"Invalid model_name '{model_name}'. Only 'altfreezing' supported.")

    detector_path = model_configs[model_name]["detector_path"]
    weights_path = model_configs[model_name]["weights_path"]
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # Load YAML config
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)

    if manual_seed is not None:
        config['manualSeed'] = manual_seed
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(manual_seed)

    # Instantiate Model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    model.eval()

    # Load Weights
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print(f"Model loaded from: {weights_path}")

    # -------------------------------------------------
    # 2) Define Face Alignment Function
    # -------------------------------------------------
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = "./training/dataset/shape_predictor_81_face_landmarks.dat"
    face_predictor = dlib.shape_predictor(predictor_path)

    def align_and_preprocess_frame(img_rgb):
        """
        Detects, aligns, and processes face before passing to the model.
        """
        resolution = config['resolution']
        mean = config['mean']
        std = config['std']
    
        face, landmarks, mask = extract_aligned_face_dlib(face_detector, face_predictor, img_rgb, res=resolution)
    
        if face is None:
            # If no face detected, resize whole image instead
            aligned_img = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        else:
            aligned_img = face  # Use aligned face
    
        pil_img = Image.fromarray(aligned_img)
        tensor_no_norm = T.ToTensor()(pil_img)
        tensor_norm = T.Normalize(mean, std)(tensor_no_norm.clone())
        return tensor_norm, tensor_no_norm

    # -------------------------------------------------
    # 3) Read and Align Video Frames
    # -------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    frames_rgb = []

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
    cap.release()

    if len(frames_rgb) == 0:
        raise ValueError(f"No frames read from {video_path}")

    # -------------------------------------------------
    # Y) Extract 4 Non-Overlapping 8-Frame Clips
    # -------------------------------------------------
    num_total_frames = len(frames_rgb)
    if num_total_frames < num_clips * num_frames:
        raise ValueError(f"Video too short: {num_total_frames} frames, needs {num_clips * num_frames}")

    clip_predictions = []
    gradcam_overlays = []
    red_percentages = []

    clip_start_idxs = np.linspace(0, num_total_frames - num_frames, num_clips, dtype=int)

    for clip_idx, start in enumerate(clip_start_idxs):
        clip_frames = frames_rgb[start:start + num_frames]

        # Preprocess Frames
        all_frames_norm = []
        all_frames_no_norm = []
       # Correct: Uses face alignment before passing frames to the model
        for frame_rgb in clip_frames:
            norm, no_norm = align_and_preprocess_frame(frame_rgb)  # Face-aligned frames
            all_frames_norm.append(norm)
            all_frames_no_norm.append(no_norm)


        # Stack Frames & Add Batch Dim
        clip_tensor_norm = torch.stack(all_frames_norm, dim=0).unsqueeze(0).to(device)
        clip_tensor_no_norm = torch.stack(all_frames_no_norm, dim=0).unsqueeze(0).to(device)
        clip_tensor_norm = clip_tensor_norm.permute(0, 2, 1, 3, 4)  # Shape: [1, C, T, H, W]

        data_dict = {'image': clip_tensor_norm}

        # -------------------------------------------------
        # 4) Run Inference on This Clip
        # -------------------------------------------------
        with torch.no_grad():
            output_dict = model(data_dict, inference=True, gradcam_mode="framewise") 
           #output_dict = model(data_dict, inference=True)
            clip_prob = output_dict['prob'].detach().cpu().numpy()[0]

        clip_predictions.append(clip_prob)

        # -------------------------------------------------
        # 5) Aggregate Grad-CAM for the Clip (One Overlay Per Clip)
        # -------------------------------------------------
        gradcam_tensor = output_dict.get("gradcam")  # Shape: (B, T, H, W)
        gradcam_folder = "/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/gradcam_n/"
        os.makedirs(gradcam_folder, exist_ok=True)
        
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        #create subfolder
        video_gradcam_folder = os.path.join(gradcam_folder, video_filename)
        os.makedirs(video_gradcam_folder, exist_ok=True)


        if gradcam_tensor is not None:
            gradcam_tensor = gradcam_tensor.squeeze(0).detach().cpu().numpy()  # Shape: (T, H, W)

            # **Choose aggregation strategy**
            if gradcam_aggregation == "mean":
                heatmap_aggregated = np.mean(gradcam_tensor, axis=0)
            elif gradcam_aggregation == "max":
                heatmap_aggregated = np.max(gradcam_tensor, axis=0)
            elif gradcam_aggregation == "last_frame":
                heatmap_aggregated = gradcam_tensor[-1]
            else:
                raise ValueError(f"Invalid Grad-CAM aggregation mode: {gradcam_aggregation}")

            # Resize heatmap & Normalize
            heatmap_resized = cv2.resize(heatmap_aggregated, (config['resolution'], config['resolution']))
            heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
            heatmap_uint8 = np.uint8(255 * heatmap_norm)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Compute Red Percentage
            def calculate_red_percentage(heatmap, red_threshold=150, non_red_threshold=100):
                blue, green, red = cv2.split(heatmap)
                red_mask = ((red >= red_threshold) & (blue <= non_red_threshold) & (green <= non_red_threshold))
                return (np.sum(red_mask) / heatmap.size) * 100.0

            red_pct = calculate_red_percentage(heatmap_color)
            red_percentages.append(red_pct)

            # Overlay on the first frame of the clip
            frame_rgb = all_frames_no_norm[0].permute(1, 2, 0).cpu().numpy()
            frame_uint8 = np.uint8(255 * frame_rgb)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            
            # Resize the original frame to the same dimensions as heatmap_color
            frame_bgr = cv2.resize(frame_bgr, (config['resolution'], config['resolution']))
            
            overlay = cv2.addWeighted(frame_bgr, 0.6, heatmap_color, 0.4, 0)
            overlay_path = os.path.join(video_gradcam_folder, f"gradcam_frame_{clip_idx:03d}.png")
            cv2.imwrite(overlay_path, overlay)
        
            # Append a tuple: (clip index, overlay file path, red percentage)
            gradcam_overlays.append((clip_idx, overlay_path, red_pct))

    # -------------------------------------------------
    # 6) Aggregate Predictions Across 4 Clips
    # -------------------------------------------------
    avg_prob = np.mean(clip_predictions)
    final_prediction = 1 if avg_prob >= 0.5 else 0  


    #  Delete to free memory
    del clip_tensor_norm
    del clip_tensor_no_norm
    del data_dict
    del output_dict
    
    return {"avg_prob": avg_prob, "final_prediction": final_prediction, "gradcam_overlays": gradcam_overlays} 


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

def is_valid_mp4(file_path):
    """Check if an MP4 file exists and can be opened."""     
    if not os.path.exists(file_path):
        return False
    cap = cv2.VideoCapture(file_path)
    is_valid = cap.isOpened()
    cap.release()
    return is_valid

def extract_base_filename(filename):
    # Remove the '.mp4' extension if present.
    if filename.endswith(".mp4"):
        filename = filename[:-4]
    # Return the first part before any underscore.
    return filename.split("_")[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on images or video with a specified model."
    )
    parser.add_argument("model_name", type=str,
                        choices=["ucf", "xception", "spsl", "altfreezing"],
                        help="Name of the model to use for inference. Choices: 'ucf', 'xception', 'spsl', and 'altfreezing'.")
    # For images, image_paths is required; for video mode, these paths are ignored.
    parser.add_argument("image_paths", nargs="*", type=str,
                        help="Paths to the images for inference. Provide one or more image paths (ignored in video mode).")
    parser.add_argument("--video", action="store_true",
                        help="If set, treat input as video; paths will be loaded from the JSON file.")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=3)

    args = parser.parse_args()

    if args.video:
        # If video paths are provided via command-line, use them.
        if args.image_paths:
            test_paths = args.image_paths
            print("Using provided video path:")
            print(test_paths)
        else:
            # ------------------------ Default JSON loading -------------------------
            with open("training/FF-FS.json", "r") as f:
                data = json.load(f)
            
            # Extract filenames from the FF-FS branch in JSON
            #FF_Fsh_test_paths = list(data["FF-FS"]["FF-FS"]["test"]["c23"].keys())
            #FF_Fsh_test_paths = list(data["Celeb-DF-v2"]["CelebDFv2_fake"]["test"].keys())
            fake_paths = list(data["FF-FS"]["FF-FS"]["test"]["c23"].keys())
            #real_paths = list(data["FF-FS"]["FF-real"]["test"]["c23"].keys())
            
            # Construct absolute file paths for videos
            #all_test_paths = ["./training/Fsh_videos/FFpp_FSh/" + filename + ".mp4" for filename in FF_Fsh_test_paths]
            #all_test_paths = ["./training/Celeb-DF-v2/Celeb-synthesis/" + filename + ".mp4" for filename in FF_Fsh_test_paths]
            fake_video_paths = ["./training/FaceShifter/FFpp_FSh/" + f + ".mp4" for f in fake_paths]
            #real_video_paths = ["./training/FaceShifter/FFpp_YT_originals/" + r + ".mp4" for r in real_paths]

            
           # Function to check if an mp4 file is valid
            def is_valid_mp4(file_path):
                if not os.path.exists(file_path):  # Check if file exists
                    return False
                cap = cv2.VideoCapture(file_path)
                is_valid = cap.isOpened()  # Check if video file can be opened
                cap.release()
                return is_valid
            
            
            # Count for debugging
            invalid_count = 0
            valid_fake_paths = []

            for path in fake_video_paths:
                if is_valid_mp4(path):
                    valid_fake_paths.append(path)
                else:
                    invalid_count += 1
            
            print("Total valid videos:", len(fake_video_paths))
            print("Number of invalid videos:", invalid_count)

            '''print("Extracted test paths before vlidation:", all_test_paths[:5])
            
            # Filter only valid mp4 files
            test_paths = [path for path in all_test_paths if is_valid_mp4(path)]
            print("Are there invalid videos? Answer:", len(all_test_paths) > len(test_paths))
            print("Number of invalid videos:", len(all_test_paths) - len(test_paths))
            print("Number of valid videos: ", len(test_paths))'''
            
            # ----------------------- End of JSON loading -----------------------------#

       
        total = 0
        correct = 0

        # Process each valid video
        for video_path in valid_fake_paths:
            result = run_inference_multiple_clips(
                model_name=args.model_name,
                video_path=video_path,
                num_frames=args.num_frames,
                num_clips=8,
                gradcam_aggregation="mean",
                cuda=True,
                manual_seed=42,
            )

            print("\nInference result for video:", os.path.basename(video_path))
            print(result)

            # Construct the textual explanation
            detection_status = "detected" if result["final_prediction"] == 1 else "not detected"
            avg_prob_percentage = round(result["avg_prob"] * 100, 2)
        
            # Identify the frame with the highest red percentage
            highest_frame = max(result["gradcam_overlays"], key=lambda x: x[2])
            highest_frame_number = highest_frame[0]
            highest_red_percentage = round(highest_frame[2] * 100, 1)
        
            explanation = (
                f"AltFreezing has {detection_status} spatial and temporal features in the video "
                f"with an average probability of {avg_prob_percentage}%. The Grad-CAM analysis highlights frame {highest_frame_number}, "
                f"which shows the highest activation with {highest_red_percentage}% red intensity."
            )

            print(explanation)

            # assume all FF-FS videos are "fake", i.e. label = 1.
            true_label = 1
            predicted_label = result["final_prediction"]

            # Update accuracy counters
            if predicted_label == true_label:
                correct += 1
            total += 1

        # After processing all videos, calculate accuracy
        print("accuracy calculaton sgarted")
        if total > 0:
            accuracy = correct / total
            print(f"\nAccuracy on {total} celeb v2 videos: {accuracy:.4f}")
        else:
            print("\nNo valid videos to calculate accuracy.")

        
