import os
import re
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import lmdb
import pickle
from tqdm import tqdm
from utils import frame_utils

class NonSequentialFramePairDataset(Dataset):
    def __init__(self, folder_paths, pair_stride, src_stride, save_dir="./pair_file_paths"):
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]
        
        self.pair_stride = pair_stride
        self.src_stride = src_stride
        self.save_dir = save_dir

        # Name for the LMDB file
        self.lmdb_filename = os.path.join(
            save_dir, f"pairs_pairstride{pair_stride}_srcstride{src_stride}.lmdb"
        )

        # Check if LMDB exists
        if not os.path.exists(self.lmdb_filename):
            # Create LMDB
            os.makedirs(save_dir, exist_ok=True)
            self._create_lmdb(folder_paths)
        
        # Open the LMDB environment
        self.env = lmdb.open(
            self.lmdb_filename,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        # Read the keys and dataset length
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
            
    def _create_lmdb(self, folder_paths):
        print("Creating LMDB file...")
        # Generate pairs
        pairs = self._generate_pairs(folder_paths)

        # Estimate the size of the database
        map_size = len(pairs) * (1024 * 1024 * 1)  # Adjust the size as needed

        env = lmdb.open(self.lmdb_filename, map_size=map_size)

        keys = []
        txn = env.begin(write=True)
        for idx, (img1_path, img2_path) in enumerate(tqdm(pairs, desc="Creating LMDB")):
            # Read images
            img1 = frame_utils.read_gen(img1_path)
            img2 = frame_utils.read_gen(img2_path)

            # Convert images to numpy arrays
            img1 = np.array(img1).astype(np.uint8)[..., :3]  # Ensure 3 channels
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            # Serialize images
            img1_bytes = img1.tobytes()
            img2_bytes = img2.tobytes()

            # Store image shapes for reconstruction
            img1_shape = img1.shape
            img2_shape = img2.shape

            # Store image paths as bytes
            img1_path_bytes = img1_path.encode('utf-8')
            img2_path_bytes = img2_path.encode('utf-8')

            # Create a unique key for each pair
            key = f"{idx:08d}".encode('ascii')  # Zero-padded index
            keys.append(key)

            # Store the data (img1 shape, img2 shape, img1 bytes, img2 bytes, img1 path, img2 path)
            data = (img1_shape, img2_shape, img1_bytes, img2_bytes, img1_path_bytes, img2_path_bytes)
            txn.put(key, pickle.dumps(data))

            # Commit every 1000 pairs
            if (idx + 1) % 1000 == 0:
                txn.commit()  # Commit current transaction
                txn = env.begin(write=True)  # Start a new transaction

        # After the loop, commit any remaining data
        txn.put(b'__len__', pickle.dumps(len(keys)))
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.commit()  # Final commit for any remaining data

        env.close()
        print("LMDB file created.")

    def _generate_pairs(self, folder_paths):
        """Generate frame pairs based on pair_stride and src_stride"""
        all_pairs = []
        for folder_path in folder_paths:
            all_files = glob(os.path.join(folder_path, '*.png'))
            pattern = re.compile(r'(\d{3,4})\.png$')
            sorted_files = sorted(
                all_files, 
                key=lambda x: int(pattern.search(os.path.basename(x)).group(1)) if pattern.search(x) else float('inf')
            )
            source_index = 0
            while source_index < len(sorted_files) - 1:
                target_index = source_index + self.pair_stride
                if target_index < len(sorted_files):
                    all_pairs.append((sorted_files[source_index], sorted_files[target_index]))
                source_index += self.src_stride
        return all_pairs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = self.keys[idx]
            data = pickle.loads(txn.get(key))
            img1_shape, img2_shape, img1_bytes, img2_bytes, img1_path_bytes, img2_path_bytes = data

            # Reconstruct images from bytes
            img1 = np.frombuffer(img1_bytes, dtype=np.uint8).reshape(img1_shape)
            img2 = np.frombuffer(img2_bytes, dtype=np.uint8).reshape(img2_shape)

            # Decode image paths
            img1_path = img1_path_bytes.decode('utf-8')
            img2_path = img2_path_bytes.decode('utf-8')

            # Convert to tensors
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        return img1, img2, (img1_path, img2_path)
