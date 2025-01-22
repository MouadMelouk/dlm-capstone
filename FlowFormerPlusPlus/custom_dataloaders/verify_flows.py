import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock, Manager

# Directories
input_dir = "/datasets/rgb/FaceForensics++/original_sequences/youtube/c23/flows/pairstride1_srcstride16"
output_dir = os.path.expandvars("$SCRATCH/Capstone/FlowFormerPlusPlus/custom_dataloaders/youtube_c23_flows_pairstride1_srcstride16")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize counters and lock for thread safety
manager = Manager()
counter_lock = Lock()
failed_folders = manager.list()
total_failed_images = manager.Value('i', 0)
total_successful_images = manager.Value('i', 0)

# Function to load images and plot them in a matrix
def process_folder(folder_name):
    global total_failed_images, total_successful_images
    folder_path = os.path.join(input_dir, folder_name)
    output_filename = os.path.join(output_dir, f"{folder_name}_matrix.png")
    images = []
    folder_failed_images = 0
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            try:
                img_path = os.path.join(folder_path, filename)
                img = mpimg.imread(img_path)
                images.append(img)
                
                # Safely increment successful image counter
                with counter_lock:
                    total_successful_images.value += 1
            except Exception as e:
                print(f"Error opening {filename} in {folder_name}: {e}")
                
                # Safely increment failed image counter
                with counter_lock:
                    total_failed_images.value += 1
                folder_failed_images += 1

    # Track if the folder had only failed images
    if len(images) == 0:
        with counter_lock:
            failed_folders.append(folder_name)
        return
    
    # Plot images in a matrix (12 images per row)
    num_images = len(images)
    rows = (num_images // 12) + (1 if num_images % 12 else 0)
    fig, axes = plt.subplots(rows, 12, figsize=(12 * 2, rows * 2))

    # Flatten axes array if images are fewer than required
    if rows == 1:
        axes = [axes]
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis("off")
        else:
            fig.delaxes(ax)  # Remove unused axes

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved {output_filename}")

# Process each subfolder in the input directory with multiple cores
if __name__ == "__main__":
    folders = [folder_name for folder_name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder_name))]
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_folder, folder): folder for folder in folders}
        
        for future in as_completed(futures):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
                with counter_lock:
                    failed_folders.append(folder)

    # Calculate total failed folders
    total_failed_folders = len(failed_folders)
    
    print(f"Total failed folders: {total_failed_folders}")
    print(f"Total failed images: {total_failed_images.value}")
    print(f"Total successful images: {total_successful_images.value}")
