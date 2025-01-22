# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys

import lmdb

sys.path.append('.')

import os
import math
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from .albu import IsotropicResize

FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']#

def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        
        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Set video_level to False
        self.video_level = False
        self.clip_size = None
        self.lmdb = False
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb:
                if len(dataset_list)>1:
                    if all_in_pool(dataset_list,FFpp_pool):
                        lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
                        self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                    else:
                        raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
                else:
                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb")
                    self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        elif mode == 'test':
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = self.collect_img_and_label_for_one_dataset(one_data)
            if self.lmdb:
                lmdb_path = os.path.join(config['lmdb_dir'], f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb')
                self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list


        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        
        self.transform = self.init_data_aug_method()
        
    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ])
        return trans
        
    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """
        Collects image and label lists by discovering .png files in the modified directory structure.
    
        Args:
            dataset_name (str): A list containing one dataset information, e.g., 'FF-DF'.
    
        Returns:
            list: A list of discovered image paths.
            list: A list of labels.
            list: A list of video names (placeholders for compatibility).
        """
        import re
        import glob
    
        pairstride = self.config.get('pairstride', 1)
        srcstride = self.config.get('srcstride', 16)
        use_flows = self.config.get('use_flows', False)

        print(f"Using pairstride {pairstride}, srcstride: {srcstride}, use_flows: {use_flows}")
    
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []
        video_name_list = []
    
        # Try to get the dataset information from the JSON file
        if not os.path.exists(self.config['dataset_json_folder']):
            self.config['dataset_json_folder'] = self.config['dataset_json_folder'].replace(
                '/Youtu_Pangu_Security_Public', '/Youtu_Pangu_Security/public'
            )
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'Dataset {dataset_name} does not exist!')
    
        # Handle special cases for compression types
        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'
    
        # Process each label in the dataset
        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            if cp is None and dataset_name in FFpp_pool:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in FFpp_pool:
                sub_dataset_info = sub_dataset_info['c40']
    
            # Process each video in the sub-dataset
            for video_name, video_info in sub_dataset_info.items():
                unique_video_name = video_info['label'] + '_' + video_name
    
                # Original label and frames
                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                label = self.config['label_dict'][video_info['label']]
                original_frame_paths = video_info['frames']
    
                # Initialize the list of paths for this video
                paths_to_use = []
    
                if use_flows:
                    # Use flows as currently done
                    discovered_paths = []
                    frame_path = original_frame_paths[0]
                    # Remove the frame filename (e.g., `frame0001.png`) from the path
                    directory = os.path.dirname(frame_path)
    
                    # Replace `/frames/` with `/flows/pairstride{}/srcstride{}/`
                    flow_dir = "/datasets/rgb/" + directory.replace(
                        "/frames/",
                        f"/flows/pairstride{pairstride}_srcstride{srcstride}/"
                    )
    
                    # Find all .png files in the flow directory
                    discovered_files = glob.glob(os.path.join(flow_dir, "*.png"))
                    if discovered_files:
                        discovered_paths.extend(discovered_files)
    
                    # If no flows are discovered, skip this video
                    if not discovered_paths:
                        print(f"No valid flows found for video {unique_video_name}. Skipping path {flow_dir}.")
                        continue
    
                    paths_to_use = discovered_paths
    
                else:
                    # Use frames corresponding to flows
                    frame_paths = []
                    #print(f"original_frame_paths: {original_frame_paths}")
                    frame_path = original_frame_paths[0]
                    # Remove the frame filename (e.g., `frame0001.png`) from the path
                    directory = os.path.dirname(frame_path)
    
                    # Replace `/frames/` with `/flows/pairstride{}/srcstride{}/`
                    flow_dir = "/datasets/rgb/" + directory.replace(
                        "/frames/",
                        f"/flows/pairstride{pairstride}_srcstride{srcstride}/"
                    )
    
                    # Find all .png files in the flow directory
                    flow_files = glob.glob(os.path.join(flow_dir, "*.png"))
                    #print(f"flow_dir: {flow_dir}")
                    #print(f"flow_files: {flow_files}")
                    if not flow_files:
                        continue
    
                    # For each flow file, extract the frame numbers and construct frame paths
                    for flow_file in flow_files:
                        # Extract the filename
                        flow_filename = os.path.basename(flow_file)
                        # Extract frame numbers using regex
                        match = re.match(r'(\d+)_(\d+)\.png', flow_filename)
                        if match:
                            frame_num1 = match.group(1)
                            frame_num2 = match.group(2)
    
                            # Construct the frame paths
                            video_identifier = os.path.basename(directory)
                            base_dir = "/datasets/rgb/" + directory.replace(
                                f"/flows/pairstride{pairstride}_srcstride{srcstride}/",
                                "/frames/"
                            )
                            frame_path1 = os.path.join(base_dir, frame_num1 + '.png')
                            frame_path2 = os.path.join(base_dir, frame_num2 + '.png')
    
                            # Add both frame paths to the list
                            frame_paths.extend([frame_path1, frame_path2])
    
                    # Remove duplicates
                    frame_paths = list(set(frame_paths))
                    if not frame_paths:
                        print(f"No valid frame paths found for video {unique_video_name}. Skipping paths {frame_paths}.")
                        continue
    
                    paths_to_use = frame_paths
    
                # Extend the label and frame paths
                label_list.extend([label] * len(paths_to_use))
                frame_path_list.extend(paths_to_use)
                video_name_list.extend([unique_video_name] * len(paths_to_use))
    
        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)
    
        return frame_path_list, label_list, video_name_list

    
    def filter_frame_paths(self, frame_paths):
        import re  # Import regular expressions module for pattern matching
        invalid_pattern = re.compile(r'\d{4}\.png$')
        valid_frame_paths = [fp for fp in frame_paths if not invalid_pattern.search(fp)]
        return valid_frame_paths

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution'] # if self.mode == "train" else self.config['resolution']
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'{self.config["rgb_dir"]}/'+file_path
            assert os.path.exists(file_path), f"{file_path} does not exist"
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)
    
    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.
    
        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.
    
        Returns:
            The augmented image, landmark, and mask.
        """
    
        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
        
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask
    
        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints', landmark)
        augmented_mask = transformed.get('mask', mask)
    
        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)
    
        # Reset the seeds to ensure different transformations for different images
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()
    
        return augmented_img, augmented_landmark, augmented_mask


    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, placeholders for landmarks and masks,
            and the image path.
        """
        # Get the image path and label
        image_path = self.data_dict['image'][index]
        label = self.data_dict['label'][index]
        #print(f"Loaded path: {image_path}")

        # Load the image
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation

        # Do Data Augmentation
        if self.mode == 'train' and self.config['use_data_augmentation']:
            image_trans, _ , _ = self.data_aug(image)
        else:
            image_trans = deepcopy(image)
        
        # To tensor and normalize
        if not no_norm:
            image_trans = self.normalize(self.to_tensor(image_trans))

        # Return placeholders for landmarks and masks
        landmark_tensor = None
        mask_tensor = None

        return image_trans, label, landmark_tensor, mask_tensor, [image_path]

    @staticmethod
    def collate_fn(batch):
        # Unzip the batch into separate lists
        images, labels, landmarks, masks, image_paths = zip(*batch)
        
        # Stack images and labels as before
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        
        # Landmarks and masks are placeholders (None)
        landmarks = None
        masks = None
    
        # Create data_dict
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        data_dict['name'] = image_paths  # Add this line
    
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/video_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
                config = config,
                mode = 'train', 
            )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        # print(iteration)
        ...
        # if iteration > 10:
        #     break
