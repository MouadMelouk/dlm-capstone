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

# Preprocessing function that mimics old test-mode preprocessing
def preprocess_image_cv2(img_path: str, cfg: dict):
    """
    - Reads an image via CV2 (BGR -> RGB).
    - Resizes to (cfg['resolution'], cfg['resolution']) using cv2.INTER_CUBIC.
    - Converts to PIL, then applies ToTensor and Normalize(mean=cfg['mean'], std=cfg['std']).
    - Returns the resulting Tensor on the correct device.
    """
    img_path = img_path
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image at {img_path}")

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    target_size = cfg['resolution']
    img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    # Convert to PIL
    pil_img = Image.fromarray(img_rgb)

    # ToTensor
    img_tensor = T.ToTensor()(pil_img)

    # Normalize
    mean = cfg['mean']  # e.g., [0.485, 0.456, 0.406]
    std = cfg['std']    # e.g., [0.229, 0.224, 0.225]
    img_tensor = T.Normalize(mean, std)(img_tensor)

    return img_tensor.unsqueeze(0).to(device)  # Shape: (1, C, H, W)


def run_inference_on_images_with_old_preprocess(
    detector_path: str,
    weights_path: str,
    image_paths: list,
    cuda: bool = True,
    manual_seed: int = None
):
    """
    Single function where we pass image paths directly (test mode) and do NO evaluation.

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

    # 6. Run inference over all images
    probabilities = []
    pred = []
    with torch.no_grad():
        for path in image_paths:
            # Preprocess each image
            input_tensor = preprocess_image_cv2(path[0], config)

            # Model forward pass
            # The model's forward might expect a dict (depending on your design).
            # Adjust below if your model forward is different:
            pathnew= os.path.join(*path[0].split('/')[-2:])
            data_dict = {'image': input_tensor, 'image_path':[[pathnew]]}
            output_dict = model(data_dict, inference=True)
            
            # Suppose 'prob' is your final classification probability in output_dict
            prob = output_dict['prob'].cpu().numpy()[0]  # shape: scalar or (1,) array
            # If your model outputs a single probability for "fake", prob might be float
            probabilities.append(float(prob))
            
            #get the hard (0/1) output
            results = model.predict_labels(data_dict)
            pred.extend(results)

    return pred

def ucf_spsl_inference(detector_path, weights_path, image_path, cuda: bool = True, manual_seed: int = None):
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

# helper function for ucf and spsl function call 
def helper_gradcam(model, image_path, output_folder="/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/gradcams"):

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


# Example usage (comment out if you just want the function in a file):
if __name__ == "__main__":
    test_paths = [
        ["/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/datasets/rgb/Celeb-DF-v1/Celeb-synthesis/frames/id0_id1_0001/048.png"], ["/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/datasets/rgb/Celeb-DF-v1/Celeb-synthesis/frames/id0_id1_0001/077.png"],
        ["/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/datasets/rgb/Celeb-DF-v1/Celeb-synthesis/frames/id0_id1_0001/126.png"]
    ]
    #Results = run_inference_on_images_with_old_preprocess(
        #detector_path="/scratch/rz2288/DeepfakeBench/training/config/detector/xception.yaml",
        #weights_path="/scratch/rz2288/DeepfakeBench/training/weights/xception_best.pth",
        #image_paths=test_paths,
        #cuda=True,
        #manual_seed=42
    #)
    #print("Results:", Results)

    # function for ucf
    # Run Grad-CAM for multiple images
    gradcam_outputs, explanations = run_gradcam_for_multiple_images(
        detector_path="/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/training/config/detector/ucf.yaml",
        weights_path="/scratch/ca2627/capstone/dlm-capstone/capstone_code/DeepfakeBench-main/training/pretrained/ucf_best.pth",
        image_paths=test_paths,
        cuda=True,
        manual_seed=42
    )

    # Print the results
    for img_path, gradcam_path, explanation in zip(test_paths, gradcam_outputs, explanations):
        print(f"Grad-CAM for {img_path[0]}: {gradcam_path}")
        print(f"Explanation: {explanation}\n")





        