import subprocess
import shlex
import ast

def wrapped_run_inference_on_images_with_old_preprocess(model_name, image_paths, cuda, manual_seed):
    """
    Runs inference on images by calling an external deepfake detection script.

    Parameters:
        model_name (str): One of "spsl", "ucf", or "xception".
        image_paths (list): List of image paths.
        cuda (bool): Whether to use CUDA (passed to the inference script if needed).
        manual_seed (int): Seed for reproducibility (passed to the inference script if needed).

    Returns:
        list: A list of tuples, each tuple containing:
            - overlay_path (str): Path to the Grad-CAM overlay image.
            - confidence (float): Softmax probability that the image is forged.
            - prediction_message (str): Verdict message from the model.
            - red_percentage (float): Percentage of red pixels in the Grad-CAM heatmap.

    Raises:
        RuntimeError: If the external command fails.
        ValueError: If the output cannot be parsed.
    """
    work_dir = "/scratch/mmm9912/Capstone/dlm-repo/capstone_code/DeepfakeBench-main/training"

    # Convert image paths to properly escaped arguments
    image_args = " ".join(shlex.quote(path) for path in image_paths)

    cmd = (
        "bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && "
        "conda activate DeepfakeBench && "
        f"python Inference_wrapper_function_ruilin.py {shlex.quote(model_name)} {image_args}'"
    )

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with error:\n{result.stderr}")

    # Extract only the relevant output
    output_lines = result.stdout.strip().split("\n")
    filtered_output = None
    for line in output_lines:
        if line.startswith("[('/"):
            filtered_output = line
            break  # Stop searching after finding the relevant output

    if filtered_output is None:
        raise ValueError(f"No valid output found in: {result.stdout}")

    try:
        parsed_output = ast.literal_eval(filtered_output)
    except Exception as e:
        raise ValueError(f"Failed to parse output: {filtered_output}") from e

    return parsed_output

# Example usage:
if __name__ == "__main__":
    model = "xception"
    images = ["./051_DF.png", "./052_DF.png", "./051_real.png", "./052_real.png"]
    try:
        results = wrapped_run_inference_on_images_with_old_preprocess(model, images, cuda=True, manual_seed=42)
        for res in results:
            print(res)
    except Exception as error:
        print(f"An error occurred: {error}")
