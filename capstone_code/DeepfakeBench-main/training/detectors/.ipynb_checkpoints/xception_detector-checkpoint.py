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
        """
        Generate Grad-CAM for the specified input_image_norm and overlay it 
        onto the non-normalized image (input_image_no_norm) with alpha blending.
    
        The functionality is unchanged from the original. 
        It's simply broken into smaller helper functions for clarity.
        """
    
        # -------------------------------------------------------------------------
        # Nested helper functions
        # -------------------------------------------------------------------------
        def _unpack_image_path(path):
            if isinstance(path, (tuple, list)):
                return path[0] if path else "Unknown"
            return path
    
        def _forward_and_backprop(image_norm, class_idx=None):
            """
            1) Forward pass on normalized image
            2) Backprop wrt the target class
            """
            # Move image to correct device
            image_norm = image_norm.to(next(self.parameters()).device)
    
            # Forward pass
            data_dict_for_cam = {'image': image_norm}
            self.eval()
            output = self(data_dict_for_cam)
    
            # Default class if not provided or out of range
            if class_idx is None or class_idx >= output['prob'].size(0):
                class_idx = 0
    
            # Backprop
            self.zero_grad()
            output['prob'][class_idx].backward(retain_graph=True)
    
            return output
    
        def _compute_gradcam_map():
            """
            3) Compute Grad-CAM map from stored gradients & activations
            """
            # weights: (batch_size=1, C, 1, 1)
            weights_local = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            gradcam = torch.sum(weights_local * self.activations, dim=1).squeeze()
            gradcam = torch.relu(gradcam)
            return gradcam.detach().cpu().numpy()
    
        def _normalize_and_resize_cam(gradcam_np, h, w):
            """
            Normalize Grad-CAM to [0..1], then resize to (h, w),
            and convert to 8-bit [0..255].
            """
            gradcam_np -= gradcam_np.min()
            gradcam_np /= (gradcam_np.max() + 1e-5)
    
            gradcam_resized = cv2.resize(gradcam_np, (w, h))
            gradcam_resized = np.uint8(255 * gradcam_resized)
            return gradcam_resized
    
        def _create_colormap(gradcam_uint8):
            """
            Turn a [H, W] uint8 array into a BGR heatmap using COLORMAP_JET.
            """
            return cv2.applyColorMap(gradcam_uint8, cv2.COLORMAP_JET)
    
        def _convert_non_norm_to_numpy(non_norm_tensor):
            """
            Convert the non-normalized input image tensor (in [0..1], shape (C,H,W))
            to a NumPy array in [0..255], shape (H,W,C). Presumably RGB.
            """
            arr = non_norm_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return arr
    
        def _build_alpha_heatmap(heatmap_bgr, gradcam_uint8):
            """
            Convert heatmap from BGR -> BGRA, then create an alpha channel 
            based on the gradcam intensity. 
            """
            heatmap_bgra_local = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2BGRA)
            alpha_float_local = gradcam_uint8.astype(np.float32) / 255.0
    
            # same logic as original: 0.7 * intensity + 0.2
            alpha_float_local = 0.7 * alpha_float_local + 0.2
            alpha_uint8 = (alpha_float_local * 255).astype(np.uint8)
            heatmap_bgra_local[..., 3] = alpha_uint8
            return heatmap_bgra_local
    
        def _blend_bgra(base_rgb_np, heatmap_bgra_local):
            """
            Manually alpha-blend BGRA heatmap onto a base RGB array.
            The base is converted to BGRA, shapes must match (H,W,4).
            Returns a final BGR image.
            """
            # Convert base from RGB to BGR
            base_bgr_local = cv2.cvtColor(base_rgb_np, cv2.COLOR_RGB2BGR)
            base_bgra_local = cv2.cvtColor(base_bgr_local, cv2.COLOR_BGR2BGRA)
    
            # Both arrays: shape (H, W, 4)
            heatmap_a = heatmap_bgra_local[..., 3:4].astype(np.float32) / 255.0
            inv_alpha = 1.0 - heatmap_a
    
            out_bgr_f = (heatmap_a * heatmap_bgra_local[..., :3].astype(np.float32)
                         + inv_alpha * base_bgra_local[..., :3].astype(np.float32))
            out_bgr_local = out_bgr_f.astype(np.uint8)
            return out_bgr_local
    
        # -------------------------------------------------------------------------
        # MAIN FUNCTION BODY
        # -------------------------------------------------------------------------
        gradcam_image_name = uuid.uuid4().hex + ".png"
        image_path = _unpack_image_path(image_path)
    
        gradcam_save_path = "/scratch/mmm9912/Capstone/FRONT_END_STORAGE/images/"
        save_path = os.path.join(gradcam_save_path, gradcam_image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        # 1) Forward pass & Backprop
        output = _forward_and_backprop(input_image_norm, target_class)
    
        # 2) Compute Grad-CAM array
        gradcam_np = _compute_gradcam_map()
    
        # 3) Normalize & resize Grad-CAM
        h, w = input_image_norm.shape[2], input_image_norm.shape[3]
        gradcam_resized = _normalize_and_resize_cam(gradcam_np, h, w)
    
        # 4) Create a BGR heatmap
        heatmap_bgr = _create_colormap(gradcam_resized)
    
        # 5) Convert non-normalized image to NumPy (RGB)
        img_no_norm_np = _convert_non_norm_to_numpy(input_image_no_norm)
    
        # 6) Convert heatmap to BGRA with alpha
        heatmap_bgra = _build_alpha_heatmap(heatmap_bgr, gradcam_resized)
    
        # 7) Blend final BGR output
        out_bgr = _blend_bgra(img_no_norm_np, heatmap_bgra)
    
        # 8) Save & return
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

