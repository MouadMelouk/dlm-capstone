import os
import re

def find_png_files_with_4digits(folder_paths):
    # Regex pattern for filenames with exactly 4 digits followed by .png
    pattern = re.compile(r'^\d{5}\.png$')
    
    for root_folder in folder_paths:
        for root, _, files in os.walk(root_folder):
            # Check for matching files
            matching_files = [file for file in files if pattern.match(file)]
            for file in matching_files:
                print(os.path.join(root, file))  # Print the full path of the matching file

# List of folder paths to search
folder_paths = [
    '/datasets/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/',
    '/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/',
    '/datasets/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/',
    '/datasets/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/',
]

# Run the search function
find_png_files_with_4digits(folder_paths)
