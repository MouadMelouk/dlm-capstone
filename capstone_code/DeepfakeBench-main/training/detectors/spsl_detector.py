'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the SPSLDetector

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
@inproceedings{liu2021spatial,
  title={Spatial-phase shallow learning: rethinking face forgery detection in frequency domain},
  author={Liu, Honggu and Li, Xiaodan and Zhou, Wenbo and Chen, Yuefeng and He, Yuan and Xue, Hui and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={772--781},
  year={2021}
}

Notes:
To ensure consistency in the comparison with other detectors, we have opted not to utilize the shallow Xception architecture. Instead, we are employing the original Xception model.
'''
import uuid
import os
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
import random
import cv2

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='spsl')
class SpslDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
        # Initialize placeholders for GradCAM
        self.activations = {}
        self.gradients = {}

        # Register hooks on middle (block2, block3) and last layers (conv3, conv4)
        self.layers_to_hook = ['block2', 'block3', 'conv3', 'conv4']
        
        for layer_name in self.layers_to_hook:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(
                lambda module, input, output, name=layer_name: self._save_activation(name, module, input, output)
            )
            layer.register_backward_hook(
                lambda module, grad_input, grad_output, name=layer_name: self._save_gradient(name, module, grad_input, grad_output)
            )

        # Variables for tracking predictions and accuracy
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def _save_activation(self, name, module, input, output):
        self.activations[name] = output

    def _save_gradient(self, name, module, grad_input, grad_output):
        self.gradients[name] = grad_output[0]


    def generate_gradcam(self, 
                         input_image_norm: torch.Tensor,
                         input_image_no_norm: torch.Tensor,
                         target_class=None, 
                         image_path=None):
        """
        Generate a Grad-CAM heatmap overlay for the specified input image.
        
        The function:
          1. Runs a forward pass (using the normalized image) and backpropagates
             with respect to the target class.
          2. Computes weighted Grad-CAM maps from several layers.
          3. Normalizes, resizes, and creates a COLORMAP_JET heatmap.
          4. Converts the original (non-normalized) image from a tensor to a NumPy RGB array.
          5. Creates an alpha channel for the heatmap and manually blends it with the original image.
          6. Saves the blended output to a predetermined directory using a UUID-generated filename.
        
        Returns:
            save_path (str): The full path where the blended image is saved.
            heatmap_bgr (np.array): The BGR heatmap without blending.
        """
        
        # -------------------------------------------------------------------------
        # 1. Set up saving details and unpack image_path if necessary
        # -------------------------------------------------------------------------
        # You can override or ignore image_path as needed – here we mimic the first code.
        if isinstance(image_path, (tuple, list)):
            image_path = image_path[0] if image_path else "Unknown"
        
        # Define base save directory and unique image name.
        base_save_dir = '/scratch/rz2288/DeepfakeBench/gradcam_output/test0210/'
        gradcam_image_name = uuid.uuid4().hex + ".png"
        save_path = os.path.join(base_save_dir, gradcam_image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # -------------------------------------------------------------------------
        # 2. Forward Pass and Backpropagation (using normalized image)
        # -------------------------------------------------------------------------
        input_image_norm = input_image_norm.to(next(self.parameters()).device)
        data_dict = {'image': input_image_norm}
        self.eval()
        output = self.forward(data_dict,inference=False)
        
        # Default to target class 0 if not provided or if out-of-range.
        if target_class is None or target_class >= output['prob'].size(0):
            target_class = 0
        
        self.zero_grad()
        output['prob'][target_class].backward(retain_graph=True)
        
        # -------------------------------------------------------------------------
        # 3. Compute the Combined (Weighted) Grad-CAM Map
        # -------------------------------------------------------------------------
        # Define the weights for each layer. Adjust these as necessary.
        layer_weights = {
            'block2': 0.15,
            'block3': 0.15,
            'conv3': 0.3,
            'conv4': 0.4
        }
        
        combined_gradcam_map = None
        # Loop over the layers you have hooked (assumed to be stored in self.layers_to_hook)
        for name in self.layers_to_hook:
            activations = self.activations[name]
            gradients = self.gradients[name]
            
            # Compute the channel-wise mean of the gradients to obtain the weights.
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            # Compute the weighted combination of activations.
            gradcam_map = torch.sum(weights * activations, dim=1).squeeze()
            gradcam_map = torch.relu(gradcam_map)
            
            # Convert to numpy and normalize.
            gradcam_map_np = gradcam_map.cpu().detach().numpy()
            gradcam_map_np -= gradcam_map_np.min()
            gradcam_map_np /= (gradcam_map_np.max() + 1e-5)
            
            # Resize to the spatial size of the input image.
            gradcam_map_resized = cv2.resize(
                gradcam_map_np, (input_image_norm.shape[3], input_image_norm.shape[2])
            )
            # Normalize again (in case resizing changed the range).
            gradcam_map_resized -= gradcam_map_resized.min()
            gradcam_map_resized /= (gradcam_map_resized.max() + 1e-5)
            
            # Apply the layer-specific weight.
            weight = layer_weights.get(name, 1.0)
            weighted_gradcam_map = gradcam_map_resized * weight
            
            if combined_gradcam_map is None:
                combined_gradcam_map = weighted_gradcam_map
            else:
                combined_gradcam_map += weighted_gradcam_map
        
        # Final normalization and conversion to 8-bit.
        combined_gradcam_map -= combined_gradcam_map.min()
        combined_gradcam_map /= (combined_gradcam_map.max() + 1e-5)
        combined_gradcam_map_uint8 = np.uint8(255 * combined_gradcam_map)
        
        # -------------------------------------------------------------------------
        # 4. Create the Heatmap
        # -------------------------------------------------------------------------
        heatmap_bgr = cv2.applyColorMap(combined_gradcam_map_uint8, cv2.COLORMAP_JET)
        
        # -------------------------------------------------------------------------
        # 5. Convert the Original (Non-normalized) Image to a NumPy Array
        # -------------------------------------------------------------------------
        # Assumes input_image_no_norm is a tensor with shape (B, C, H, W) in range [0,1].
        img_no_norm_np = input_image_no_norm[0].detach().cpu().numpy().transpose(1, 2, 0)
        img_no_norm_np = np.clip(img_no_norm_np * 255.0, 0, 255).astype(np.uint8)
        
        # -------------------------------------------------------------------------
        # 6. Build the Alpha Heatmap and Blend with the Original Image
        # -------------------------------------------------------------------------
        # Convert the heatmap to BGRA (adds an alpha channel).
        heatmap_bgra = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2BGRA)
        # Create an alpha channel from the Grad-CAM intensity (scaled as in the original logic).
        alpha_float = combined_gradcam_map_uint8.astype(np.float32) / 255.0
        alpha_float = 0.7 * alpha_float + 0.2  # Adjust transparency as desired.
        alpha_uint8 = (alpha_float * 255).astype(np.uint8)
        heatmap_bgra[..., 3] = alpha_uint8
        
        # Convert the original image (assumed RGB) to BGRA.
        base_bgr = cv2.cvtColor(img_no_norm_np, cv2.COLOR_RGB2BGR)
        base_bgra = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2BGRA)
        
        # Manually blend using the alpha channel.
        heatmap_alpha = heatmap_bgra[..., 3:4].astype(np.float32) / 255.0
        inv_alpha = 1.0 - heatmap_alpha
        blended_bgr = (heatmap_alpha * heatmap_bgra[..., :3].astype(np.float32) +
                       inv_alpha * base_bgra[..., :3].astype(np.float32)).astype(np.uint8)
        
        # -------------------------------------------------------------------------
        # 7. Save and Return
        # -------------------------------------------------------------------------
        cv2.imwrite(save_path, blended_bgr)
        return save_path, heatmap_bgr


    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

        # To get a good performance, use the ImageNet-pretrained Xception model
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}

        # remove conv1 from state_dict
        conv1_data = state_dict.pop('conv1.weight')

        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model from {}'.format(config['pretrained']))

        # copy on conv1
        # let new conv1 use old param to balance the network
        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)  # average across the RGB channels
        backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)  # repeat the averaged weights across the 4 new channels
        logger.info('Copy conv1 from pretrained model')
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict, phase_fea) -> torch.tensor:
        features = torch.cat((data_dict['image'], phase_fea), dim=1)
        return self.backbone.features(features)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
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
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the phase features
        phase_fea = self.phase_without_amplitude(data_dict['image'])
        # bp
        features = self.features(data_dict, phase_fea)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        if inference:
            self.prob.append(pred_dict['prob'].detach().squeeze().cpu().numpy())
            self.label.append(data_dict['label'].detach().squeeze().cpu().numpy())
            # deal with acc
            _, prediction_class = torch.max(pred, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

        return pred_dict

    def phase_without_amplitude(self, img):
        # Convert to grayscale
        gray_img = torch.mean(img, dim=1, keepdim=True) # shape: (batch_size, 1, 256, 256)
        # Compute the DFT of the input signal
        X = torch.fft.fftn(gray_img,dim=(-1,-2))
        #X = torch.fft.fftn(img)
        # Extract the phase information from the DFT
        phase_spectrum = torch.angle(X)
        # Create a new complex spectrum with the phase information and zero magnitude
        reconstructed_X = torch.exp(1j * phase_spectrum)
        # Use the IDFT to obtain the reconstructed signal
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X,dim=(-1,-2)))
        # reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X))
        return reconstructed_x
    
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
            outputs = self.forward(data_dict, inference=False)
            prob = outputs['prob']  # Probability of being 'fake'
            binary_preds = (prob >= threshold)
            predictions_str = [
                "SPSL model detected forgery." if is_fake else "SPSL model did not detect forgery."
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
                "SPSL model detected forgery" if is_fake else "SPSL model did not detect forgery"
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