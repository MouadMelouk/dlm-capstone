'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the XceptionDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
'''

import os
import cv2
import uuid
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='xception')
class XceptionDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0

        self.activations = None
        self.gradients = None

        self.backbone.conv4.register_forward_hook(self._save_activation)
        self.backbone.conv4.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]


    def generate_gradcam(self, input_image, target_class=None, image_path=None):    
        detector_type = "Xception"  # Just for reference
        gradcam_save_path = "/scratch/mmm9912/Capstone/FRONT_END_STORAGE/images/"
        gradcam_image_name = uuid.uuid4().hex + ".png"
        
        # Extract the actual path if image_path is a tuple containing a list
        if isinstance(image_path, tuple) and isinstance(image_path[0], list) and isinstance(image_path[0][0], str):
            image_path = image_path[0][0]
        elif not isinstance(image_path, str):
            raise TypeError("Expected image_path to be a string or a tuple containing a list with a string.")
    
        # Construct full save path
        save_path = os.path.join(gradcam_save_path, gradcam_image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        # Move the input image to the correct device
        input_image = input_image.to(next(self.parameters()).device)
        data_dict = {'image': input_image}
        
        self.eval()
        output = self(data_dict)
    
        # Choose target class
        if target_class is None or target_class >= output['prob'].size(0):
            target_class = 0
    
        self.zero_grad()
        output['prob'][target_class].backward(retain_graph=True)
    
        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        gradcam_map = torch.sum(weights * self.activations, dim=1).squeeze()
        gradcam_map = torch.relu(gradcam_map)
    
        # Normalize the Grad-CAM heatmap
        gradcam_map = gradcam_map.cpu().detach().numpy()
        gradcam_map -= gradcam_map.min()
        gradcam_map /= (gradcam_map.max() + 1e-5)
    
        # Resize and convert to 8-bit image
        gradcam_map_resized = cv2.resize(gradcam_map, (input_image.shape[3], input_image.shape[2]))
        gradcam_map_resized = np.uint8(255 * gradcam_map_resized)
    
        # Apply colormap
        heatmap = cv2.applyColorMap(gradcam_map_resized, cv2.COLORMAP_JET)
    
        # ---------------------------------------------------------------------------- #
        #                 Overlay heatmap onto the *original* input image
        # ---------------------------------------------------------------------------- #
        # 1. Convert the input_image tensor to a NumPy array for visualization.
        #    Assuming input_image is in range [0, 1] or [-1, +1], you may want to
        #    reverse any normalization you applied. Below is a simple approach
        #    if your image is already in [0, 1]. Adjust as needed.
        #
        #    If you have a batch dimension, use [0] to pick the first image.
        # ---------------------------------------------------------------------------- #
        # Example with no normalization reversion (assuming input is [0..1]):
    
        img_np = input_image[0].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        img_np = (img_np * 255).astype(np.uint8)  # scale to [0..255]
    
        # ---------------------------------------------------------------------------- #
        # 2. Blend (overlay) the original image and the heatmap using cv2.addWeighted
        # ---------------------------------------------------------------------------- #
        # Note that 'heatmap' is in BGR (because of cv2) and 'img_np' is likely in RGB
        # if loaded that way. If your input was also in BGR, this is fine; otherwise,
        # you might need to convert img_np to BGR. Adjust as needed.
        # 
        # alpha = 0.6, beta = 0.4 -> these control how strong the heatmap overlay is.
        # ---------------------------------------------------------------------------- #
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5,0)
    
        # Save the overlay image
        cv2.imwrite(save_path, overlay)
    
        return save_path, heatmap  # Optionally return the overlay if needed


    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        #print(config['pretrained'])
        state_dict = torch.load("./weights/xception_best.pth")
        #(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256
    
    def classifier(self, features: torch.tensor, get_embeddings=False):
        return self.backbone.classifier(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        overall_loss = loss
        loss_dict = {'overall': overall_loss, 'cls': loss,}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        self.video_names = []
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False, get_embeddings=False) -> dict:
        # Get the features by backbone
        features = self.features(data_dict)
        # Get the prediction (and embeddings if requested)
        if get_embeddings:
            pred, embeddings = self.classifier(features, get_embeddings=True)
        else:
            pred = self.classifier(features)
        # Get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # Build the prediction dict
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        if get_embeddings:
            pred_dict['embeddings'] = embeddings
        return pred_dict


    def predict_labels(self, data_dict: dict, threshold=0.5) -> list:

        def calculate_red_percentage(heatmap, red_threshold=150, non_red_threshold=100):
            """
            Calculates the percentage of red pixels in the given heatmap.
            
            Parameters:
                heatmap (numpy.ndarray): The Grad-CAM heatmap (BGR format).
                red_threshold (int): Minimum intensity for a pixel to be considered "red".
                non_red_threshold (int): Maximum intensity for blue/green channels to reduce false positives.
            
            Returns:
                float: Percentage of red pixels in the image (0 to 100).
            """
            # Extract B, G, R channels
            blue_channel, green_channel, red_channel = cv2.split(heatmap)
        
            # Define red pixels: High red value & lower blue/green to avoid yellow/orange
            red_mask = (red_channel >= red_threshold) & (blue_channel <= non_red_threshold) & (green_channel <= non_red_threshold)
        
            # Compute percentage of red pixels
            total_pixels = heatmap.shape[0] * heatmap.shape[1]
            red_pixels = np.sum(red_mask)
            
            return (red_pixels / total_pixels) * 100  # Convert to percentage
        """
        Predict hard labels and generate Grad-CAM heatmaps for each input image.
        Returns a list where each element is a tuple containing:
            (gradcam_heatmap, prediction_string)
        """
        self.eval()
        combined_results = []
        
        
        with torch.enable_grad():
            outputs = self.forward(data_dict, inference=True)
            prob = outputs['prob']  # Probability of being 'fake'
            binary_preds = (prob >= threshold)
            predictions_str = [
                "General model detected forgery" if is_fake else "General model did not detect forgery"
                for is_fake in binary_preds
            ]
    
            batch_size = data_dict['image'].size(0)
            
            for i in range(batch_size):
                # Extract individual image
                input_image = data_dict['image'][i].unsqueeze(0)
                input_path = data_dict['image_path'][i][0]
                
                # Generate Grad-CAM - consider setting target_class=1 for consistent visualization
                overlay_path,heatmap = self.generate_gradcam(
                    input_image,
                    target_class=None,  # Default to class 0 visualization
                    image_path=input_path
                )
                
                # Pair the heatmap with its corresponding prediction
                #combined_results.append((heatmap, predictions_str[i]))

                # Compute percentage of red pixels
                percentage_red = calculate_red_percentage(heatmap)
    
                # Append the results: (heatmap, prediction string, percentage red)
                combined_results.append((overlay_path, prob, predictions_str[i], percentage_red))
            
        return combined_results

