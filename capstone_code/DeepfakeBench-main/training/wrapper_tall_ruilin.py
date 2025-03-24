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


# Function to extract frames remains unchanged
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


def run_inference_on_images_with_cv2_preprocess(
    detector_path: str,
    weights_path: str,
    image_paths: list,
    cuda: bool = True,
    manual_seed: int = None,
    running_inference: bool = True,
    model = None,
    config = None,
    device = 'cpu'
):

    def preprocess_images_cv2(img_paths: list, cfg: dict):

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
            predictor_path = './preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'
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
                    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
                    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
                    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
                    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
                    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
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

        
        normalized_frames = []
        non_normalized_frames = []
        all_crop_coords = []
        for img_path in img_paths:
            img_tensor, img_tensor_no_norm, crop_coords = preprocess_image_cv2(img_path, cfg)
            normalized_frames.append(img_tensor)
            non_normalized_frames.append(img_tensor_no_norm)
            all_crop_coords.append(crop_coords)
        # Stack to get shape [num_frames, C, H, W] and add batch dimension
        normalized_frames = torch.stack(normalized_frames, dim=0).unsqueeze(0)
        non_normalized_frames = torch.stack(non_normalized_frames, dim=0).unsqueeze(0)
        return normalized_frames, non_normalized_frames, all_crop_coords

    # Run inference over all sets of image paths
    probabilities = []
    pred = []
    with torch.no_grad():
        for paths in image_paths:
            # Preprocess using CV2/dlib
            input_tensor_norm, input_tensor_no_norm, crop_coords_list = preprocess_images_cv2(paths, config)

            # Move tensors to device
            input_tensor_norm = input_tensor_norm.to(device)
            input_tensor_no_norm = input_tensor_no_norm.to(device)

            # Store paths (using last two folder names for display)
            path_list = [[os.path.join(*p.split('/')[-2:]) for p in paths]]

            data_dict = {
                'image': input_tensor_norm,             # normalized frames
                'image_no_norm': input_tensor_no_norm,    # non-normalized frames
                'crop_coords': crop_coords_list,          # crop coordinates per frame
                'image_path': path_list
            }

            output_dict = model(data_dict, inference=running_inference)

            # For example, get classification probability from the output
            prob = output_dict['prob'].cpu().numpy()[0]
            probabilities.append(float(prob))

            # Hard predictions from the model
            results = model.predict_labels(data_dict)
            pred.extend(results)

    return pred

# --- The rest of your code (e.g., run_inference_on_videos) remains largely the same ---
def run_inference_on_videos(
    number_grids: int,
    detector_paths: str,
    weights_paths: str,
    video_paths: list,
    cuda: bool = True,
    manual_seed: int = None,
    runninginference: bool = True,
    base_path_for_frames="outputframes_0222_1"
):
    video_path_for_inference = []
    
    for path in video_paths:
        # Extract the file name without extension to use as an ID
        file_name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(base_path_for_frames, file_name)
        os.makedirs(save_path, exist_ok=True)
        video_path_for_inference.append(save_path)
        extract_frames(num_parts=number_grids, video_path=path, output_dir=save_path)

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

    All_results = []
    for basepath in video_path_for_inference:
        res = []
        for i in range(number_grids):
            test_paths = [[]]
            for j in range(4):
                frame_name = f"{i*4+j:03d}.jpg"
                frame_path = os.path.join(basepath, frame_name)
                test_paths[0].append(frame_path)
            Result = run_inference_on_images_with_cv2_preprocess(
                detector_path=detector_paths,
                weights_path=weights_paths,
                image_paths=test_paths,
                cuda=cuda,
                manual_seed=42,
                running_inference=runninginference,
                model=model,
                config=config,
                device=device
            )
            res.append(Result)
        
        total_score = sum([res[i][0][1] for i in range(number_grids)]) / number_grids
        if total_score > 0.5:
            elem = (total_score, "Tall model detected forgery.")
        else:
            elem = (total_score, "Tall model did not detect forgery.")
        All_results.append([elem, res])
        
    return All_results


    

import json
# Example usage (comment out if you just want the function in a file):
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference on videos with a specified model.")
    parser.add_argument("model_name", type=str, choices=["tall"], 
                        help="Name of the model to use for inference. Choices: tall")
    parser.add_argument("video_paths", nargs="+", type=str, 
                        help="Paths to the videos for inference. Provide one or more video paths.")

    args = parser.parse_args()

    # Wrap each image path in a list, maintaining the original nested structure
    all_test_paths = args.video_paths

    
    # Path to your JSON file
    #json_file_path = "/scratch/rz2288/DeepfakeBench/preprocessing/dataset_json/Celeb-DF-v1.json"
    
    #with open(json_file_path, "r") as f:
    #    data = json.load(f)
    
    # Extract filenames from JSON
    #FF_Fsh_test_paths = list(data["Celeb-DF-v1"]["CelebDFv1_fake"]["test"].keys())
    
    # Construct absolute file paths
    #all_test_paths = ["/scratch/rz2288/DeepfakeBench/datasets/rgb/Celeb-DF-v1/Celeb-synthesis/" + filename + ".mp4" for filename in FF_Fsh_test_paths]

    res = run_inference_on_videos(
        number_grids=8,
        detector_paths="/scratch/rz2288/DeepfakeBench/training/config/detector/tall.yaml",
        weights_paths="/scratch/rz2288/DeepfakeBench/training/weights/tall_trainFF_testCDF.pth",
        video_paths=all_test_paths,
        cuda=True,
        manual_seed=42,
        runninginference = False 
        # only false for spsl+tall due to original code integrated label and accuracy calculation in spsl_detector class function
    )
    for i in res:
        print(i)



