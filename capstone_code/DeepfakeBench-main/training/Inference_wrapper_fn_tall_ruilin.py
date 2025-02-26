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
import torch
import yaml
import dlib
from facenet_pytorch import MTCNN


# Function to extract frames
def extract_frames(video_path, output_dir, num_parts=8, frames_per_part=4):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_segment = total_frames // num_parts

    saved_files = []
    frame_number = 0

    for part in range(num_parts):
        start_frame = part * frames_per_segment
        end_frame = (part + 1) * frames_per_segment

        # Randomly select a starting position within the segment
        random_start = random.randint(start_frame, end_frame - frames_per_part)

        # Set the video capture to the random starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_start)

        for i in range(frames_per_part):
            success, frame = cap.read()
            if not success:
                break

            frame_path = os.path.join(output_dir, f"{frame_number:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_files.append(frame_path)
            frame_number += 1

    cap.release()
    return saved_files

def run_inference_on_images_with_mtcnn_preprocess(
    detector_path: str,
    weights_path: str,
    image_paths: list,
    cuda: bool = True,
    manual_seed: int = None,
    running_inference: bool = True,
    model = None,
    mtcnn = None,
    config = None,
    device = 'cpu'
):
    """
    Modified function to use MTCNN for face detection and preprocessing.
    Extracts the largest face bounding box and adds 30% padding to each side.

    Args:
        detector_path (str): Path to the detector YAML (config).
        weights_path (str): Path to the model checkpoint weights.
        image_paths (list): List of lists; each sub-list contains absolute image file paths
                            (frames) to run inference on.
        cuda (bool): If True and CUDA is available, uses GPU.
        manual_seed (int): Optional manual seed for reproducibility.
        running_inference (bool): If True, model is called in inference mode.

    Returns:
        list: A list of predicted labels (or probabilities) for each sequence in `image_paths`.
    """

    def expand_bbox(bbox, img_width, img_height, padding_ratio=0.3):
        """
        Expands the bounding box by a given padding ratio (30% by default).
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Calculate padding
        pad_x = int(width * padding_ratio)
        pad_y = int(height * padding_ratio)

        # Expand bounding box
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_width, x2 + pad_x)
        y2 = min(img_height, y2 + pad_y)

        return [x1, y1, x2, y2]

    def preprocess_image_with_mtcnn(img_paths: list, cfg: dict):
        """
        Preprocesses a list of image paths using MTCNN for face detection and alignment.
        Extracts the largest face bounding box and adds 30% padding.

        Returns:
            (tensor_norm, tensor_no_norm, list_of_crop_coords)
            Each tensor shape: [1, num_frames, 3, H, W].
        """
        normalized_frames = []
        non_normalized_frames = []
        all_crop_coords = []

        for img_path in img_paths:
            # Read BGR image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError(f"Could not read image at {img_path}")

            # BGR -> RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            boxes, _ = mtcnn.detect(img_rgb)

            if boxes is not None and len(boxes) > 0:
                # Select the largest face (maximum area bounding box)
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                largest_idx = np.argmax(areas)
                bbox = boxes[largest_idx]

                # Expand the bounding box by 30%
                img_height, img_width, _ = img_rgb.shape
                expanded_bbox = expand_bbox(bbox, img_width, img_height, padding_ratio=0.3)

                # Crop the face region
                x1, y1, x2, y2 = map(int, expanded_bbox)
                face_crop = img_rgb[y1:y2, x1:x2]

                # Resize to target resolution
                target_size = cfg['resolution']
                face_crop_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

                # Convert to PIL image
                pil_img = Image.fromarray(face_crop_resized)

                # Store crop coordinates
                all_crop_coords.append([x1, y1, x2, y2])
            else:
                # No face detected; fallback to resizing the entire image
                target_size = cfg['resolution']
                face_crop_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
                pil_img = Image.fromarray(face_crop_resized)
                all_crop_coords.append(None)

            # ToTensor
            img_tensor_no_norm = T.ToTensor()(pil_img)  # (C, H, W) in [0..1]
            # Normalize
            img_tensor = T.Normalize(cfg['mean'], cfg['std'])(img_tensor_no_norm.clone())

            normalized_frames.append(img_tensor)
            non_normalized_frames.append(img_tensor_no_norm)

        # Stack to get shape [num_frames, C, H, W]
        normalized_frames = torch.stack(normalized_frames, dim=0)
        non_normalized_frames = torch.stack(non_normalized_frames, dim=0)

        # Add batch dimension => [1, num_frames, C, H, W]
        normalized_frames = normalized_frames.unsqueeze(0)
        non_normalized_frames = non_normalized_frames.unsqueeze(0)

        return normalized_frames, non_normalized_frames, all_crop_coords

    # Run inference over all sets of image paths
    probabilities = []
    pred = []
    with torch.no_grad():
        for paths in image_paths:
            # Preprocess
            input_tensor_norm, input_tensor_no_norm, crop_coords_list = preprocess_image_with_mtcnn(paths, config)

            # Move to device
            input_tensor_norm = input_tensor_norm.to(device)
            input_tensor_no_norm = input_tensor_no_norm.to(device)

            # Store paths
            path_list = [[os.path.join(*p.split('/')[-2:]) for p in paths]]

            data_dict = {
                'image': input_tensor_norm,             # normalized frames
                'image_no_norm': input_tensor_no_norm,  # non-normalized frames
                'crop_coords': crop_coords_list,        # one per frame
                'image_path': path_list
            }

            output_dict = model(data_dict, inference=running_inference)

            # Example: classification probability or score in output_dict
            prob = output_dict['prob'].cpu().numpy()[0]
            probabilities.append(float(prob))

            # Hard predictions from your model
            results = model.predict_labels(data_dict)
            pred.extend(results)

    return pred


def run_inference_on_videos(
    number_grids:int,
    detector_paths: str,
    weights_paths: str,
    video_paths: list,
    cuda: bool = True,
    manual_seed: int = None,
    runninginference: bool = True,
    base_path_for_frames = "outputframes_0222_1"
):
    
    video_path_for_inference = []
    
    for path in video_paths:
        # Extract the file name with extension
        file_name_with_extension = os.path.basename(path)
        # Remove the extension to get the ID
        file_name = os.path.splitext(file_name_with_extension)[0]
        save_path = os.path.join(base_path_for_frames,file_name)
        os.makedirs(save_path, exist_ok=True)
        video_path_for_inference.append(save_path)
        extract_frames(num_parts=number_grids,video_path = path, output_dir=save_path)


    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # 1. Load YAML config
    with open(detector_paths, 'r') as f:
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

    # 4. Load checkpoint weights
    ckpt = torch.load(weights_paths, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print(f"Model loaded from: {weights_paths}")

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device=device)

    All_results=[]
    for basepath in video_path_for_inference:
        res=[]
        for i in range(number_grids):
            test_paths=[[]]
            for j in range(4):
                frame_name=f"{i*4+j:03d}.jpg"
                frame_path = os.path.join(basepath, frame_name)
                test_paths[0].append(frame_path)
            Result = run_inference_on_images_with_mtcnn_preprocess(
            detector_path=detector_paths,
            weights_path=weights_paths,
            image_paths=test_paths,
            cuda=cuda,
            manual_seed=42,
            running_inference = runninginference, # only false for spsl+tall due to original code integrated label and accuracy calculation in spsl_detector class function
                model = model,
                mtcnn = mtcnn,
                config=config,
                device = device
        )
            res.append(Result)
        
        total_score=0
        for i in range(number_grids):
            total_score+=res[i][0][1]

        total_score = total_score/number_grids
        if total_score> 0.5:
            elem = (total_score, "Tall model detected forgery.")
        else:
            elem = (total_score,"Tall model did not detect forgery.")
        All_results.append([elem,res])
        
    return All_results

    


# Example usage (comment out if you just want the function in a file):
if __name__ == "__main__":
    test_paths = [
    "/scratch/rz2288/DeepfakeBench/datasets/rgb/Celeb-DF-v1/Celeb-synthesis/id1_id0_0007.mp4",
    "/scratch/rz2288/DeepfakeBench/datasets/rgb/Celeb-DF-v1/Celeb-synthesis/id2_id0_0008.mp4",
    "/scratch/rz2288/DeepfakeBench/datasets/rgb/Celeb-DF-v1/Celeb-real/id1_0007.mp4",
    "/scratch/rz2288/DeepfakeBench/datasets/rgb/Celeb-DF-v1/Celeb-real/id3_0001.mp4"
    ]
    res = run_inference_on_videos(
        number_grids=8,
        detector_paths="/scratch/rz2288/DeepfakeBench/training/config/detector/tall.yaml",
        weights_paths="/scratch/rz2288/DeepfakeBench/training/weights/tall_trainFF_testCDF.pth",
        video_paths=test_paths,
        cuda=True,
        manual_seed=42,
        runninginference = False 
        # only false for spsl+tall due to original code integrated label and accuracy calculation in spsl_detector class function
    )
    for i in res:
        print(i)


