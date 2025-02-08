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

    
    def generate_gradcam(self, 
                       input_image_norm: torch.Tensor,
                       input_image_no_norm: torch.Tensor,
                       target_class=None, 
                       image_path=None):
        detector_type = "Xception"
        gradcam_save_path = "/scratch/mmm9912/Capstone/FRONT_END_STORAGE/images/"
        gradcam_image_name = uuid.uuid4().hex + ".png"
    
        # Unpack image_path
        if isinstance(image_path, (tuple, list)):
            image_path = image_path[0] if image_path else "Unknown"
  
        save_path = os.path.join(gradcam_save_path, gradcam_image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        # -------------------------------
        # 1) Forward pass on normalized image
        # -------------------------------
        input_image_norm = input_image_norm.to(next(self.parameters()).device)
        data_dict_for_cam = {'image': input_image_norm}
        self.eval()
        output = self(data_dict_for_cam)
  
        if target_class is None or target_class >= output['prob'].size(0):
            target_class = 0
  
        # -------------------------------
        # 2) Backprop
        # -------------------------------
        self.zero_grad()
        output['prob'][target_class].backward(retain_graph=True)
  
        # -------------------------------
        # 3) Grad-CAM
        # -------------------------------
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        gradcam_map = torch.sum(weights * self.activations, dim=1).squeeze()
        gradcam_map = torch.relu(gradcam_map)
  
        gradcam_map = gradcam_map.detach().cpu().numpy()
        gradcam_map -= gradcam_map.min()
        gradcam_map /= (gradcam_map.max() + 1e-5)
  
        # Resize Grad-CAM to match the input resolution
        height, width = input_image_norm.shape[2], input_image_norm.shape[3]
        gradcam_map_resized = cv2.resize(gradcam_map, (width, height))
        gradcam_map_resized = np.uint8(255 * gradcam_map_resized)
    
        # Apply a colormap in BGR
        heatmap_bgr = cv2.applyColorMap(gradcam_map_resized, cv2.COLORMAP_JET)
    
        # -------------------------------
        # 4) Convert the *non-normalized* image to NumPy
        #    (in [0..255], shape (H,W,3)), presumably in RGB
        # -------------------------------
        img_no_norm_np = input_image_no_norm[0].detach().cpu().numpy().transpose(1, 2, 0)
        img_no_norm_np = np.clip(img_no_norm_np * 255.0, 0, 255).astype(np.uint8)
    
        # -------------------------------
        # 5) Convert heatmap to BGRA with alpha = f(Grad-CAM intensity)
        # -------------------------------
        heatmap_bgra = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2BGRA)
    
        alpha_float = gradcam_map_resized.astype(np.float32) / 255.0
        alpha_float = 0.7 * alpha_float + 0.2  # keep final in [0..1]
        heatmap_bgra[..., 3] = (alpha_float * 255).astype(np.uint8)
    
        # -------------------------------
        # 6) Convert your base image to BGRA for manual alpha blending
        #    If it's actually RGB, convert. If it's already BGR, skip the COLOR_RGB2BGR step.
        # -------------------------------
        base_bgr = cv2.cvtColor(img_no_norm_np, cv2.COLOR_RGB2BGR)
        base_bgra = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2BGRA)
    
        # Make sure shapes match (H, W, 4)
        # Now do alpha blending manually:
        heatmap_a = (heatmap_bgra[..., 3:4].astype(np.float32) / 255.0)  # alpha in [0..1]
        inv_alpha = (1.0 - heatmap_a)
    
        # Weighted combination per pixel => out_rgb is BGR channels actually
        out_bgr_f = (heatmap_a * heatmap_bgra[..., :3].astype(np.float32)
                   + inv_alpha * base_bgra[..., :3].astype(np.float32))
        out_bgr = out_bgr_f.astype(np.uint8)
    
        # Save & return
        cv2.imwrite(save_path, out_bgr)
        return save_path, heatmap_bgr



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
            # same as before
            blue_channel, green_channel, red_channel = cv2.split(heatmap)
            red_mask = ((red_channel >= red_threshold) &
                        (blue_channel <= non_red_threshold) &
                        (green_channel <= non_red_threshold))
            total_pixels = heatmap.shape[0] * heatmap.shape[1]
            red_pixels = np.sum(red_mask)
            return (red_pixels / total_pixels) * 100.0

        self.eval()
        combined_results = []
    
        with torch.enable_grad():
            outputs = self.forward(data_dict, inference=True)
            prob = outputs['prob']  # Probability of being 'fake'
            binary_preds = (prob >= threshold)
            predictions_str = [
                "General model detected forgery." if is_fake else "General model did not detect forgery."
                for is_fake in binary_preds
            ]
    
            batch_size = data_dict['image'].size(0)
            
            for i in range(batch_size):
                # Normalized tensor (for forward pass / backprop)
                input_image_norm = data_dict['image'][i].unsqueeze(0)
                # Non-normalized tensor (for visualization)
                input_image_no_norm = data_dict['image_no_norm'][i].unsqueeze(0)
    
                input_path = data_dict['image_path'][i][0]
    
                # Generate Grad-CAM
                overlay_path, heatmap = self.generate_gradcam(
                    input_image_norm,
                    input_image_no_norm,
                    target_class=None,  # or 1 if your "fake" label is 1
                    image_path=input_path
                )
    
                # Example of computing a metric from heatmap
                percentage_red = calculate_red_percentage(heatmap)
    
                combined_results.append(
                    (overlay_path, float(prob[i]), predictions_str[i], percentage_red)
                )
        return combined_results

        
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

