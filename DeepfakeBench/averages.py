import os
import numpy as np

# List of paths to process
paths = [
    "/datasets/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/flows/pairstride1_srcstride16/",
    "/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/flows/pairstride1_srcstride16/",
    "/datasets/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/flows/pairstride1_srcstride16/",
    "/datasets/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/flows/pairstride1_srcstride16/",
    "/datasets/rgb/FaceForensics++/original_sequences/youtube/c23/flows/pairstride1_srcstride16/",
    "/datasets/rgb/FaceForensics++/original_sequences/actors/c23/flows/pairstride1_srcstride16/"
]

# Function to compute stats for a given path
def compute_frame_stats(path):
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    subfolders = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    frame_counts = []

    for subfolder in subfolders:
        frame_count = len([f for f in os.listdir(subfolder) if f.endswith(".png")])
        frame_counts.append(frame_count)

    if frame_counts:
        avg_frames = np.mean(frame_counts)
        min_frames = np.min(frame_counts)
        avg_frames_int = int(np.floor(avg_frames))
        num_avg_or_more = sum(1 for count in frame_counts if count >= avg_frames_int)
        num_min = sum(1 for count in frame_counts if count == min_frames)
        num_eight_or_more = sum(1 for count in frame_counts if count >= 8)

        print(f"Path: {path}")
        print(f"  Average frames: {avg_frames:.2f}")
        print(f"  Minimum frames: {min_frames}")
        print(f"  Folders with frames >= average (rounded down): {num_avg_or_more}")
        print(f"  Folders with frames == minimum: {num_min}")
        print(f"  Folders with frames >= 8: {num_eight_or_more}")
        print(f"  Total folders: {len(subfolders)}")
    else:
        print(f"Path: {path} contains no valid subfolders with PNG files.")

# Process each path
for path in paths:
    compute_frame_stats(path)
