'''
# author: Kangran Zhao
# email: kangranzhao@link.cuhk.edu.cn
# date: 2023-0822
# description: Class for the TALLDetector

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
@inproceedings{xu2023tall,
  title={TALL: Thumbnail Layout for Deepfake Video Detection},
  author={Xu, Yuting and Liang, Jian and Jia, Gengyun and Yang, Ziming and Zhang, Yanhao and He, Ran},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22658--22668},
  year={2023}
}
'''

import logging

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from detectors import DETECTOR
from einops import rearrange
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.hub import load_state_dict_from_url
import cv2
import os
import uuid
import numpy as np
import torch.nn.functional as F

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='tall')
class TALLDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.model = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        model_kwargs = dict(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
                            mlp_ratio=config['mlp_ratio'], patch_size=config['patch_size'],
                            window_size=config['window_size'], depths=config['depths'],
                            num_heads=config['num_heads'], ape=config['ape'],
                            thumbnail_rows=config['thumbnail_rows'], drop_rate=config['drop_rate'],
                            drop_path_rate=config['drop_path_rate'], use_checkpoint=False, bottleneck=False,
                            duration=config['clip_size'])
        default_cfg = {
            'url': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
            'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
            'crop_pct': .9, 'interpolation': 'bicubic',
            'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
            'first_conv': 'patch_embed.proj', 'classifier': 'head', }
        backbone = SwinTransformer(img_size=config['resolution'], **model_kwargs)
        backbone.default_cfg = default_cfg
        load_pretrained(backbone, num_classes=config['num_classes'], in_chans=model_kwargs.get('in_chans', 3),
                        filter_fn=_conv_filter, img_size=config['resolution'], pretrained_window_size=7, pretrained_model='')

        return backbone

    def generate_attentionmap(self, 
                              input_image_norm: torch.Tensor,
                              input_image_no_norm: torch.Tensor,
                              target_class=None, 
                              image_path=None):
        """
        Generate an attention map by averaging attention weights across
        all blocks in the last layer of the Swin Transformer.
        """
        import cv2
        import os
        import uuid
        import math
        import numpy as np
        import torch.nn.functional as F
        
        self.model.eval()
        device = next(self.model.parameters()).device
        input_image_norm = input_image_norm.to(device)
        
        # List to capture attention weights from each block in the last layer.
        attention_weights_list = []
        
        # Identify the last layer of the Swin model
        last_layer_idx = len(self.model.layers) - 1
        last_layer = self.model.layers[last_layer_idx]
        
        # Hook function to capture attention weights for each block
        def make_hook_fn():
            def hook_fn(module, input, output):
                # We assume the attention module stores its computed weights in "attn_weights".
                if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                    attention_weights_list.append(module.attn_weights.detach())
            return hook_fn
        
        # Register a forward hook on each block's attention module in the last layer
        hooks = []
        for block in last_layer.blocks[-5:]:
            attention_module = block.attn
            h = attention_module.register_forward_hook(make_hook_fn())
            hooks.append(h)
        
        # Forward pass with the full video tensor
        with torch.no_grad():
            data_dict_forward = {'image': input_image_norm}
            _ = self.forward(data_dict_forward)
        
        # Remove all hooks
        for h in hooks:
            h.remove()
        
        # Check if we captured any attention weights
        if not attention_weights_list:
            raise ValueError("No attention weights captured from the last layer.")
        
        # 1) Stack all attention weights from each block, shape each is (num_windows*B, num_heads, N, N)
        # 2) Then average across blocks (dim=0)
        attn = torch.stack(attention_weights_list, dim=0).mean(dim=0)  # shape: (num_windows*B, num_heads, N, N)
        
        B_, num_heads, N, _ = attn.shape
        window_size = int(math.sqrt(N))
        
        # Instead of averaging everything, let's do a "mean over heads" and then
        # a "max across tokens" to preserve contrast.
        attn_avg = attn.mean(dim=1)                # shape: (B_, N, N)
        attn_map = attn_avg.max(dim=-1).values     # shape: (B_, N)
        attn_map = attn_map.view(B_, window_size, window_size)  # (B_, ws, ws)
        
        # Get the patch grid resolution from the last layer
        H_layer, W_layer = last_layer.input_resolution
        num_windows_h = H_layer // window_size
        num_windows_w = W_layer // window_size
        batch_size = input_image_norm.size(0)
        
        # Reassemble the windowed attention maps into one full map
        attn_map = attn_map.view(batch_size, num_windows_h, num_windows_w, window_size, window_size)
        attn_map = attn_map.permute(0, 1, 3, 2, 4).contiguous()  # shape: (B, H_layer, W_layer)
        attn_map = attn_map.view(batch_size, H_layer, W_layer)
        
        # Convert the non–normalized video (5D) into the thumbnail
        B, T, C, H_orig, W_orig = input_image_no_norm.shape
        thumbnail_input = input_image_no_norm.view(B, T * C, H_orig, W_orig)
        thumbnail = self.model.create_thumbnail(thumbnail_input)  # (B, 3, H_thumb, W_thumb)
        H_thumb, W_thumb = thumbnail.shape[2], thumbnail.shape[3]
        
        # Upsample the attention map to match the thumbnail size
        attn_upsampled = F.interpolate(
            attn_map.unsqueeze(1).float(), 
            size=(H_thumb, W_thumb), 
            mode='bicubic', 
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Normalize the upsampled attention map to [0, 1]
        attn_upsampled = (attn_upsampled - attn_upsampled.min()) / (
            attn_upsampled.max() - attn_upsampled.min() + 1e-8
        )
        heatmap = np.uint8(255 * attn_upsampled)
        
        # Create a colored heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert the thumbnail to a NumPy image (BGR)
        thumb_img = thumbnail.squeeze().permute(1, 2, 0).cpu().numpy()
        thumb_img = (thumb_img * 255).astype(np.uint8)
        thumb_img = cv2.cvtColor(thumb_img, cv2.COLOR_RGB2BGR)
        
        # Overlay the heatmap
        superimposed = cv2.addWeighted(thumb_img, 0.5, heatmap_colored, 0.5, 0)
        
        # Save to disk
        attention_image_path = uuid.uuid4().hex + ".png"
        save_dir = '/scratch/rz2288/DeepfakeBench/gradcam_output/test0215_2/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, attention_image_path)
        
        if np.isnan(superimposed).any() or np.isinf(superimposed).any():
            print("Error: Image contains NaN or Inf values")
        success = cv2.imwrite(save_path, superimposed)
        if not success:
            print("Error: Failed to save image!")
            blank_image = np.zeros((448, 448, 3), dtype=np.uint8)
            cv2.imwrite(save_path, blank_image)
        if superimposed is None or superimposed.size == 0:
            print("Error: superimposed is empty!")
        
        return save_path, heatmap_colored
    

    
    def generate_attentionmap_singlelayer(self, 
                              input_image_norm: torch.Tensor,
                              input_image_no_norm: torch.Tensor,
                              target_class=None, 
                              image_path=None):
        """
        Generate an attention map using the Swin Transformer's attention weights
        from a full video input (e.g., (1, 4, 3, 224, 224)) and overlay it on the
        combined (thumbnail) original image.
        
        The function:
          1. Registers a hook on a chosen attention module to capture attention weights.
          2. Forwards the full video tensor through the detector.
          3. Processes the captured attention weights into a spatial map.
          4. Converts the non–normalized video (5D) into a 4D tensor and calls create_thumbnail.
          5. Upsamples the attention map to the thumbnail size, applies a colormap,
             overlays the heatmap on the thumbnail, and saves the result.
        
        Returns:
            tuple: (save_path, heatmap_colored)
        """
        import cv2
        import os
        import uuid
        import numpy as np
        import torch.nn.functional as F
    
        # Ensure the model is in eval mode and on the correct device.
        #print(self.model)
        self.model.eval()
        device = next(self.model.parameters()).device
        input_image_norm = input_image_norm.to(device)
    
        # List to capture attention weights.
        attention_weights = []
    
        # Hook function to capture attention weights.
        def hook_fn(module, input, output):
            # We assume the attention module stores its computed weights in "attn_weights".
            if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                attention_weights.append(module.attn_weights.detach())
    
        # Select a module from a specific block.
        layer_idx = 2
        block_idx = -1
        selected_layer = self.model.layers[layer_idx]
        selected_block = selected_layer.blocks[block_idx]
        attention_module = selected_block.attn
    
        # Register the forward hook.
        handle = attention_module.register_forward_hook(hook_fn)
    
        # Forward pass with the full video tensor.
        with torch.no_grad():
            data_dict_forward = {'image': input_image_norm}
            _ = self.forward(data_dict_forward)
    
        # Remove the hook.
        handle.remove()
    
        if not attention_weights:
            raise ValueError("No attention weights captured.")
    
        # Process the captured attention weights.
        # Expected shape: (num_windows * B, num_heads, N, N)
        attn = attention_weights[0]
        B_, num_heads, N, _ = attn.shape
        window_size = int(math.sqrt(N))
    
        # Average over heads and then average across tokens (diagonals).
        #attn_avg = attn.mean(dim=1)           # Shape: (B_, N, N)
        #attn_map = attn_avg.mean(dim=-1)        # Shape: (B_, N)
        attn_avg = attn.mean(dim=1)                # shape: (B_, N, N)
        attn_map = attn_avg.max(dim=-1).values     # shape: (B_, N)
        attn_map = attn_map.view(B_, window_size, window_size)  # (B_, ws, ws)
    
        # Get the patch grid resolution from the selected layer.
        H_layer, W_layer = selected_layer.input_resolution  # e.g., (56, 56)
        num_windows_h = H_layer // window_size
        num_windows_w = W_layer // window_size
        batch_size = input_image_norm.size(0)
    
        # Reshape and reassemble the windowed attention maps into one full map.
        attn_map = attn_map.view(batch_size, num_windows_h, num_windows_w, window_size, window_size)
        attn_map = attn_map.permute(0, 1, 3, 2, 4).contiguous()  # Now shape: (B, H_layer, W_layer)
        attn_map = attn_map.view(batch_size, H_layer, W_layer)
    
        # --- Combine frames for visualization ---
        # The detector's create_thumbnail expects a 4D tensor (B, (th*tw*c), H, W)
        # so we first merge the temporal dimension and channel dimension.
        B, T, C, H_orig, W_orig = input_image_no_norm.shape
        thumbnail_input = input_image_no_norm.view(B, T * C, H_orig, W_orig)
        # Now call create_thumbnail (which will also perform resizing if necessary)
        thumbnail = self.model.create_thumbnail(thumbnail_input)  # shape: (B, 3, H_thumb, W_thumb)
        H_thumb, W_thumb = thumbnail.shape[2], thumbnail.shape[3]
    
        # Upsample the attention map to match the thumbnail size.
        attn_upsampled = F.interpolate(
            attn_map.unsqueeze(1).float(), 
            size=(H_thumb, W_thumb), 
            mode='bicubic', 
            align_corners=False
        ).squeeze().cpu().numpy()
    
        # Normalize the upsampled attention map to [0, 1] and convert to uint8.
        attn_upsampled = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)
        heatmap = np.uint8(255 * attn_upsampled)
    
        # Create a colored heatmap.
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
        # Convert the thumbnail (non–normalized video) to a NumPy image in BGR format.
        thumb_img = thumbnail.squeeze().permute(1, 2, 0).cpu().numpy()
        thumb_img = (thumb_img * 255).astype(np.uint8)
        thumb_img = cv2.cvtColor(thumb_img, cv2.COLOR_RGB2BGR)
    
        # Overlay the heatmap onto the thumbnail.
        superimposed = cv2.addWeighted(thumb_img, 0.5, heatmap_colored, 0.5, 0)
    
        attention_image_path = uuid.uuid4().hex + ".png"
        save_dir = '/scratch/rz2288/DeepfakeBench/gradcam_output/test0222_11/'  # Update as needed.
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, attention_image_path)
        if np.isnan(superimposed).any() or np.isinf(superimposed).any():
            print("Error: Image contains NaN or Inf values")
        success = cv2.imwrite(save_path, superimposed)
    
        return save_path, heatmap_colored

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        # STIL requires the input with the shape of (n, t*c, h, w), where n is the batch_size, t is num_segment
        #print(data_dict['image'].shape)
        bs, t, c, h, w = data_dict['image'].shape

        inputs = data_dict['image'].view(bs, t * c, h, w)
        pred = self.model(inputs)
        return pred

    def classifier(self, features: torch.tensor):
        pass

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].long()
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
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the prediction by backbone
        pred = self.features(data_dict)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': prob}
        if inference:
            self.prob.extend(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.extend(
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


    def predict_labels(self, data_dict: dict, threshold=0.5) -> list:
        """
        For each video in the batch, forward the entire set of frames (e.g. shape (1,4,3,224,224))
        and generate a single attention overlay (thumbnail) for the video.
        Then, compute the red percentage from the heatmap and return the results.
        """
        def calculate_red_percentage(heatmap, red_threshold=150, non_red_threshold=100):
            # Compute percentage of strongly red pixels in the heatmap.
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
                "Tall model detected forgery." if is_fake else "Tall model did not detect forgery."
                for is_fake in binary_preds
            ]
    
            batch_size = data_dict['image'].size(0)
            # data_dict['image'] and data_dict['image_no_norm'] are assumed to be 5D: (B, T, C, H, W)
            for i in range(batch_size):
                # Forward the entire video (all frames together)
                input_video_norm = data_dict['image'][i].unsqueeze(0)       # shape (1, 4, 3, 224, 224)
                input_video_no_norm = data_dict['image_no_norm'][i].unsqueeze(0)  # shape (1, 4, 3, 224, 224)
    
                # Use the provided image path (or generate one if needed)
                input_path = data_dict['image_path'][i][0]
    
                # Generate the attention overlay for the entire video
                overlay_path, heatmap = self.generate_attentionmap_singlelayer(
                    input_video_norm,
                    input_video_no_norm,
                    target_class=None,  # Not used here
                    image_path=input_path
                )
    
                percentage_red = calculate_red_percentage(heatmap)
                combined_results.append(
                    (overlay_path, float(prob[i]), predictions_str[i], percentage_red)
                )
        return combined_results

    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.attn_weights = None  # Add this line to store attention weights

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        #attn = self.softmax(attn)
        self.attn_weights = attn  # Store attention weights after softmax

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, bottleneck=False, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward_attn(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        return x

    def forward_mlp(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_attn, x)
        else:
            x = self.forward_attn(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_mlp, x)
        else:
            x = x + self.forward_mlp(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 bottleneck=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 bottleneck=bottleneck if i == depth - 1 else False,
                                 use_checkpoint=use_checkpoint)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, duration=8, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, thumbnail_rows=1, bottleneck=False, **kwargs):
        super().__init__()

        self.duration = duration  # 4
        self.num_classes = num_classes  # 2
        self.num_layers = len(depths)  # [2, 2, 18, 2]
        self.embed_dim = embed_dim  # 128
        self.ape = ape  # True
        self.patch_norm = patch_norm  # False
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio  # 4 = default
        self.thumbnail_rows = thumbnail_rows  # 2

        self.img_size = img_size  # 224
        self.window_size = [window_size for _ in depths] if not isinstance(window_size, list) else window_size
        # self.image_mode = True                # [14, 14, 14, 7]

        self.frame_padding = self.duration % thumbnail_rows  # 0
        if self.frame_padding != 0:
            self.frame_padding = self.thumbnail_rows - self.frame_padding
            self.duration += self.frame_padding

        # split image into non-overlapping patches
        thumbnail_dim = (thumbnail_rows, self.duration // thumbnail_rows)  # (2, 2)
        thumbnail_size = (img_size * thumbnail_dim[0], img_size * thumbnail_dim[1])

        self.patch_embed = PatchEmbed(
            img_size=(img_size, img_size), patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  # 16
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution  # [56, 56]

        # absolute position embedding
        if self.ape:  # True
            self.frame_pos_embed = nn.Parameter(torch.zeros(1, self.duration, embed_dim))
            trunc_normal_(self.frame_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=self.window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               bottleneck=bottleneck)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'frame_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def create_thumbnail(self, x):
        # import pdb;pdb.set_trace()
        input_size = x.shape[-2:]
        if input_size != to_2tuple(self.img_size):
            x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.thumbnail_rows, c=3)
        return x

    def pad_frames(self, x):
        frame_num = self.duration - self.frame_padding
        x = x.view((-1, 3 * frame_num) + x.size()[2:])
        x_padding = torch.zeros((x.shape[0], 3 * self.frame_padding) + x.size()[2:]).cuda()
        x = torch.cat((x, x_padding), dim=1)
        assert x.shape[1] == 3 * self.duration, 'frame number %d not the same as adjusted input size %d' % (
            x.shape[1], 3 * self.duration)

        return x

    # need to find a better way to do this, maybe torch.fold?
    def create_image_pos_embed(self):
        img_rows, img_cols = self.patches_resolution  # (56, 56)
        _, _, T = self.frame_pos_embed.shape  # (1, 4, embed)
        rows = img_rows // self.thumbnail_rows  # 28
        cols = img_cols // (self.duration // self.thumbnail_rows)  # 28
        img_pos_embed = torch.zeros(img_rows, img_cols, T).cuda()  # [56, 56, embed]
        for i in range(self.duration):
            r_indx = (i // self.thumbnail_rows) * rows
            c_indx = (i % self.thumbnail_rows) * cols
            img_pos_embed[r_indx:r_indx + rows, c_indx:c_indx + cols] = self.frame_pos_embed[0, i]

        return img_pos_embed.reshape(-1, T)  # [56*56, embed]

    def forward_features(self, x):
        if self.frame_padding > 0:
            x = self.pad_frames(x)
        else:
            x = x.view((-1, 3 * self.duration) + x.size()[2:])

        x = self.create_thumbnail(x)
        x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')  # [B, 3, 224, 224]

        x = self.patch_embed(x)  # [B, 56*56, embed]
        if self.ape:
            img_pos_embed = self.create_image_pos_embed()
            x = x + img_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, img_size=224, num_patches=196,
                    pretrained_window_size=7, pretrained_model="", strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    if len(pretrained_model) == 0:
        # state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
        state_dict = load_state_dict_from_url(cfg['url'], map_location='cpu')
    else:
        try:
            state_dict = load_state_dict(pretrained_model)['model']
        except:
            state_dict = load_state_dict(pretrained_model)

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            _logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight


    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:  # and len(pretrained_model) == 0:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict['model'][classifier_name + '.weight']
        del state_dict['model'][classifier_name + '.bias']
        strict = False
    '''
    ## Resizing the positional embeddings in case they don't match
    if img_size != cfg['input_size'][1]:
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed
   '''

    # remove window_size related parameters
    window_size = (model.window_size)[0]
    print(pretrained_window_size, window_size)

    new_state_dict = state_dict['model'].copy()
    for key in state_dict['model']:
        if 'attn_mask' in key:
            del new_state_dict[key]

        if 'relative_position_index' in key:
            del new_state_dict[key]

        # resize it
        if 'relative_position_bias_table' in key:
            pretrained_table = state_dict['model'][key]
            pretrained_table_size = int(math.sqrt(pretrained_table.shape[0]))
            table_size = int(math.sqrt(model.state_dict()[key].shape[0]))
            if pretrained_table_size != table_size:
                table = pretrained_table.permute(1, 0).view(1, -1, pretrained_table_size, pretrained_table_size)
                table = nn.functional.interpolate(table, size=table_size, mode='bilinear')
                table = table.view(-1, table_size * table_size).permute(1, 0)
                new_state_dict[key] = table

    for key in model.state_dict():
        if 'bottleneck_norm' in key:
            attn_key = key.replace('bottleneck_norm', 'norm1')
            # print (key, attn_key)
            new_state_dict[key] = new_state_dict[attn_key]

    print('loading weights....')
    ## Loading the weights
    model.load_state_dict(new_state_dict, strict=False)


def _conv_filter(state_dict, patch_size=4):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
