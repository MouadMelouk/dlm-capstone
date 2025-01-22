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

import os
import cv2
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
        layers_to_hook = ['block2', 'block3', 'conv3', 'conv4']
        
        for layer_name in layers_to_hook:
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

    def generate_gradcam(self, input_image, target_class=None, image_path="./gradcam_outputs"):
        detector_type = "SPSL_avg_b2_15_b3_15_c3_30_c4_40"
        datasets_base_path = "/scratch/mh6117/MLLM_CS_AIGC_data/datasets/rgb/"

        input_image = input_image.to(next(self.parameters()).device)
        data_dict = {'image': input_image}
        self.eval()
        output = self(data_dict)
    
        if target_class is None or target_class >= output['prob'].size(0):
            target_class = 0
    
        self.zero_grad()
        output['prob'][target_class].backward(retain_graph=True)
    
        # Define weights for each layer
        layer_weights = {
            'block2': 0.15,
            'block3': 0.15,
            'conv3': 0.3,
            'conv4': 0.4
        }
    
        combined_gradcam_map = None
        for name in self.layers_to_hook:
            activations = self.activations[name]
            gradients = self.gradients[name]
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            gradcam_map = torch.sum(weights * activations, dim=1).squeeze()
            gradcam_map = torch.relu(gradcam_map)
    
            gradcam_map = gradcam_map.cpu().detach().numpy()
            gradcam_map -= gradcam_map.min()
            gradcam_map /= (gradcam_map.max() + 1e-5)
    
            gradcam_map_resized = cv2.resize(
                gradcam_map, (input_image.shape[3], input_image.shape[2])
            )
            # Normalize each map individually
            gradcam_map_resized -= gradcam_map_resized.min()
            gradcam_map_resized /= (gradcam_map_resized.max() + 1e-5)
    
            # Apply weight to the gradcam_map
            weight = layer_weights.get(name, 1.0)  # Default weight is 1.0 if not specified
            weighted_gradcam_map = gradcam_map_resized * weight
    
            if combined_gradcam_map is None:
                combined_gradcam_map = weighted_gradcam_map
            else:
                combined_gradcam_map += weighted_gradcam_map
    
        # Normalize the combined map
        combined_gradcam_map -= combined_gradcam_map.min()
        combined_gradcam_map /= (combined_gradcam_map.max() + 1e-5)
        combined_gradcam_map = np.uint8(255 * combined_gradcam_map)
    
        heatmap = cv2.applyColorMap(combined_gradcam_map, cv2.COLORMAP_JET)
    
        image_name = os.path.basename(image_path)
        save_name = f"{os.path.splitext(image_name)[0]}_{detector_type}_GradCAM_heatmap.png"
        save_path = os.path.join(datasets_base_path, os.path.dirname(image_path), save_name)
    
        cv2.imwrite(save_path, heatmap)
    
        return heatmap
    
    def generate_embedding(self, data, save_path):
        """
        Generates and saves the embedding for a given input.
        
        Args:
            data (torch.Tensor): The input image tensor.
            save_path (str): The path to save the embedding.
        """
        # Ensure the model is in evaluation mode
        self.eval()
    
        # Disable gradient computation for inference
        with torch.no_grad():
            # Generate phase features
            phase_fea = self.phase_without_amplitude(data)
            # Prepare data_dict for model input
            data_dict = {'image': data}
            # Extract features (concatenate image and phase features)
            features = self.features(data_dict, phase_fea)
            # Get prediction and embeddings from classifier
            _, embedding = self.classifier(features, get_embeddings=True)
    
        # Convert the embedding to a numpy array
        embedding_np = embedding.cpu().numpy()
    
        # Save the embedding to the specified path
        np.save(save_path, embedding_np)
    
        # Print its shape and path
        #print(f"Saved embedding to {save_path}, shape: {embedding_np.shape}")


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
    
    def classifier(self, features: torch.tensor, get_embeddings=False):
        return self.backbone.classifier(features, get_embeddings=get_embeddings)

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
        
    def forward(self, data_dict: dict, inference=False, get_embeddings=False) -> dict:
        # Get the phase features
        phase_fea = self.phase_without_amplitude(data_dict['image'])
        # Extract features
        features = self.features(data_dict, phase_fea)
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
            
        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
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