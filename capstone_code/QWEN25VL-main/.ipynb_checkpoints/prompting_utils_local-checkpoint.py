import cv2
import os
import uuid
import json

import os
from supabase import create_client, Client

def extract_k_frames(video_path, k, frames_dir):
    """
    Extracts k equally spaced frames from a video and returns a list of tuples.
    Each tuple contains:
       (original_frame_index, saved_image_file_path)

    Parameters:
        video_path (str): Path to the input video file.
        k (int): Number of frames to extract.
        frames_dir (str): Directory where the extracted frames will be saved.

    Returns:
        list of (int, str): A list of tuples, where each tuple has:
            - The original frame index in the video.
            - The absolute path to the saved frame image.
    """
    # Create the frames directory (UUID-based) if it doesn't already exist
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[extract_k_frames] Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < k:
        cap.release()
        raise ValueError(f"[extract_k_frames] Video has only {frame_count} frames, but {k} are required.")

    # Compute k equally spaced frame indices
    frame_indices = [int(i * frame_count / k) for i in range(k)]
    saved_frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"[extract_k_frames] Failed to read frame at index {idx}")

        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(frames_dir, filename)
        cv2.imwrite(file_path, frame)
        saved_frames.append((idx, file_path))

    cap.release()
    return saved_frames

def parse_llm_response(response_str):
    """
    Parses the JSON response from the LLM and returns a structured dictionary.

    Parameters:
        response_str (str): The raw JSON string returned by the LLM.

    Returns:
        dict: A structured dictionary with keys:
            - "direct_answer" (str or None): Direct response from the LLM.
            - "consult_expert" (dict or None): If deepfake analysis is needed, contains:
                - "expert_model_name" (str)
                - "video_path" (str)
                - "number_of_frames" (int)
    """
    try:
        response_data = json.loads(response_str)
        
        direct_answer = response_data.get("direct_answer_to_frontend", None)
        consult_expert = response_data.get("consult_expert_model", None)

        # Ensure the expert model structure is correctly formatted
        if consult_expert and all(
            key in consult_expert for key in ["expert_model_name", "video_path", "number_of_frames"]
        ):
            return {
                "direct_answer": direct_answer if direct_answer else None,
                "consult_expert": {
                    "expert_model_name": consult_expert["expert_model_name"],
                    "video_path": consult_expert["video_path"],
                    "number_of_frames": consult_expert["number_of_frames"],
                },
            }
        else:
            return {
                "direct_answer": direct_answer if direct_answer else None,
                "consult_expert": None,
            }
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from LLM")


def PLACEHOLDERwrapped_run_inference_on_images_with_old_preprocess(model_name, image_paths, cuda, manual_seed):
    """
    PLACEHOLDER function call to run inference on images using expert DL models.
    In prod, this function would run the deepfake detection models.
    
    Parameters:
        model_name (str): One of "spsl", "ucf", or "xception".
        image_paths (list): List of image paths.
        cuda (bool): Whether to use CUDA.
        manual_seed (int): Seed for reproducibility.
    
    Returns:
        list: A list of tuples, each tuple containing:
            - overlay_path (str): Path to the Grad-CAM overlay image.
            - confidence (float): Softmax probability that the image is forged.
            - prediction_message (str): Verdict message from the model.
            - red_percentage (float): Percentage of red pixels in the Grad-CAM heatmap.
    """
    overlay_path = "/scratch/mmm9912/Capstone/FRONT_END_STORAGE/images/ca4227e5f59643179b25ba59c0483b9b.png"
    confidence = 0.75
    prediction_message = f"{model_name.upper()} model detected forgery."
    red_percentage = 10.0

    return [(overlay_path, confidence, prediction_message, red_percentage) for _ in image_paths]

import subprocess
import shlex
import shutil
import ast

def wrapped_run_inference_on_images_with_old_preprocess(
    model_name,
    image_paths,
    cuda,
    manual_seed,
    gradcams_dir
):
    """
    Runs inference on images by calling an external deepfake detection script.
    The script writes Grad-CAM overlays (e.g., to /scratch/mmm9912/Capstone/FRONT_END_STORAGE/images/).
    Here, we parse the script's output, then MOVE those overlays to `gradcams_dir`.

    Parameters:
        model_name (str): One of "spsl", "ucf", or "xception".
        image_paths (list): List of extracted frame paths.
        cuda (bool): If True, tries GPU; otherwise CPU.
        manual_seed (int): Reproducibility seed.
        gradcams_dir (str): The final directory to store Grad-CAM images after moving them.

    Returns:
        list of tuples: (new_overlay_path, confidence, message, red_percentage)
    """

    os.makedirs(gradcams_dir, exist_ok=True)

    # Convert image paths into command-line arguments
    image_args = " ".join(shlex.quote(path) for path in image_paths)

    # This command calls your external script, e.g., Inference_wrapper_function_ruilin.py
    work_dir = "/scratch/mmm9912/Capstone/dlm-repo/capstone_code/DeepfakeBench-main/training"
    cmd = (
        "bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && "
        "conda activate DeepfakeBench && "
        f"python Inference_wrapper_function_ruilin.py {shlex.quote(model_name)} "
        f"{image_args}'"
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
        raise RuntimeError(f"[wrapped_run_inference_on_images] Command failed:\n{result.stderr}")

    # The external script presumably prints something that starts with [('/...
    output_lines = result.stdout.strip().split("\n")
    filtered_output = None
    for line in output_lines:
        if line.startswith("[('/"):
            filtered_output = line
            break

    if filtered_output is None:
        raise ValueError(
            "[wrapped_run_inference_on_images] No valid output found in external script. "
            f"Full stdout:\n{result.stdout}"
        )

    # Attempt to parse the bracketed list
    try:
        parsed_output = ast.literal_eval(filtered_output)
    except Exception as e:
        raise ValueError(
            f"[wrapped_run_inference_on_images] Failed to parse output: {filtered_output}"
        ) from e

    updated_results = []
    # Move each Grad-CAM overlay from its original location to gradcams_dir
    for i, (overlay_path, confidence, prediction_message, red_percentage) in enumerate(parsed_output, start=1):
        # E.g., gradcams_dir/1.png, 2.png, etc.
        new_name = f"{i}.png"
        new_path = os.path.join(gradcams_dir, new_name)

        # Make sure the directory exists
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # Move the file
        try:
            shutil.move(overlay_path, new_path)
        except Exception as move_err:
            raise RuntimeError(
                f"[wrapped_run_inference_on_images] Could not move {overlay_path} -> {new_path}: {move_err}"
            )

        updated_results.append((new_path, confidence, prediction_message, red_percentage))

    return updated_results


def consult_expert_model(video_path, k, model_name, cuda, manual_seed):
    """
    Extracts frames from the video, runs inference on each extracted frame,
    and returns a list where:
      - The first element is a summary of the analysis (string).
      - The subsequent elements are dictionaries (one per frame), each containing:
          {
            "frame_index": <int>,
            "overlay_path": <str>,
            "confidence": <float>,
            "message": <str>,
            "red_percentage": <float or None>
          }

    Example return shape:
    [
      "The assistant has made an expert function call to UCF. ...",
      {
        "frame_index": 0,
        "overlay_path": "/path/to/gradcams/abc123/1.png",
        "confidence": 23.517408967018127,
        "message": "UCF model did not detect forgery.",
        "red_percentage": None
      },
      ...
    ]

    The summary string is formatted as:
      "The assistant has made an expert function call to {MODEL}. The frames analyzed by {MODEL}
       were {frame_indices_str}. In frame {idx1} forgery was detected with confidence {conf1}%, ...
       The average confidence was {avg_confidence}%. The area of the image that potentially
       contains adversarial artifacts comprise of X%..."

    Parameters:
      video_path (str): Path to the input video file
      k (int): Number of frames to extract
      model_name (str): Name of the expert model ("spsl", "ucf", "xception", etc.)
      cuda (bool): Whether to use CUDA
      manual_seed (int): Seed for reproducibility

    Returns:
      list:
        [ summary_string, dict_for_frame_1, dict_for_frame_2, ... ]
    """

    import uuid

    # 1) Build a unique ID for directories
    inference_uuid = uuid.uuid4().hex[:8]
    frames_dir = f"/scratch/mmm9912/Capstone/dlm-repo/capstone_code/QWEN25VL-main/storage/frames/{inference_uuid}"
    gradcams_dir = f"/scratch/mmm9912/Capstone/dlm-repo/capstone_code/QWEN25VL-main/storage/gradcams/{inference_uuid}"

    print(f"[consult_expert_model] Starting expert analysis with model '{model_name}'.")
    print(f"[consult_expert_model] Storing Grad-CAM overlays in: {gradcams_dir}")

    # 2) Extract frames
    frames = extract_k_frames(video_path, k, frames_dir)
    image_paths = [frame_path for (_, frame_path) in frames]
    print(f"[consult_expert_model] Number of frames extracted: {len(frames)}")

    # 3) Inference with direct saving of Grad-CAMs
    inference_results = wrapped_run_inference_on_images_with_old_preprocess(
        model_name, image_paths, cuda, manual_seed, gradcams_dir
    )
    print(f"[consult_expert_model] Inference complete. # of results returned: {len(inference_results)}")

    # 4) Combine each frame's info with its inference result
    combined_results = []
    for (frame_index, _), (overlay_path, confidence, prediction_message, red_percentage) in zip(frames, inference_results):
        conf_percent = confidence * 100.0
        combined_results.append({
            "frame_index": frame_index,
            "overlay_path": overlay_path,
            "confidence": conf_percent,
            "message": prediction_message,
            "red_percentage": red_percentage if "detected forgery" in prediction_message.lower() else None
        })

    # 5) Build the summary string
    frame_indices_str = ", ".join(str(item["frame_index"]) for item in combined_results)
    details = []
    red_values = []
    total_confidence = 0.0

    for item in combined_results:
        total_confidence += item["confidence"]
        if item["red_percentage"] is not None:
            details.append(
                f"In frame {item['frame_index']} forgery was detected "
                f"with confidence {item['confidence']:.1f}%"
            )
            red_values.append(f"{item['red_percentage']:.1f}%")
        else:
            details.append(
                f"In frame {item['frame_index']} no forgery was detected "
                f"with confidence {item['confidence']:.1f}%"
            )

    details_str = ", ".join(details)
    avg_confidence = total_confidence / len(combined_results) if combined_results else 0.0

    red_str = ""
    if red_values:
        red_str = (
            " The area of the image that potentially contains adversarial artifacts "
            f"comprise of {', '.join(red_values)} of each frame, respectively."
        )

    summary = (
        f"The assistant has made an expert function call to {model_name.upper()}. "
        f"The frames analyzed by {model_name.upper()} were {frame_indices_str}. "
        f"{details_str}. The average confidence was {avg_confidence:.1f}%." + red_str
    )

    return [summary] + combined_results



def handle_llm_response(response_str):
    """
    Parses the LLM's JSON response and, if needed, calls the deepfake detection model.

    Parameters:
        response_str (str): The raw JSON string returned by the LLM.

    Returns:
        - If an expert model is consulted: The result from `consult_expert_model()`.
        - Otherwise: The direct answer from the LLM.
    """
    parsed_response = parse_llm_response(response_str)

    # If an expert model consultation is required
    if parsed_response["consult_expert"]:
        expert_model = parsed_response["consult_expert"]["expert_model_name"]
        video_path = parsed_response["consult_expert"]["video_path"]
        num_frames = parsed_response["consult_expert"]["number_of_frames"]

        # Perform the expert model consultation
        return consult_expert_model(video_path, num_frames, expert_model, cuda=True, manual_seed=42)

    # Otherwise, return the direct answer
    return parsed_response["direct_answer"]

# Example usage:
#llm_response = '{\n  "direct_answer_to_frontend": "",\n  "consult_expert_model": {\n    "expert_model_name": "ucf",\n    "video_path": "/scratch/mmm9912/Capstone/FRONT_END_STORAGE/videos/8be0d76e-3dba-4970-9e85-49122ca690c8.mp4",\n    "number_of_frames": 4\n  }\n}'

#result = handle_llm_response(llm_response)
#print(result)

def format_user_or_assistant_message(role, prompt):
    # role: "assistant" or "user"
    # prompt: string
    return {
        "role": role,
        "content": json.dumps({
            "direct_answer_to_frontend": prompt,
            "consult_expert_model": None
        }, indent=2)
    }

#text = "Well, yeah, but what exactly do you do? What is the range of your possibilities?"
#formatted_message = format_user_or_assistant_message(text)
#print(formatted_message)


def format_expert_message(expert_feedback):
    return {
        "role": "system",
        "content": expert_feedback
    }

#text = "Well, yeah, but what exactly do you do? What is the range of your possibilities?"
#formatted_message = format_expert_message(text)
#print(formatted_message)

# Import the required functions from the supabase_wrapper module.
# These functions interact with the online database.

def get_latest_message_info():
    """
    Retrieves the latest message's ID along with its conversation ID from the database.
    """
    # Fetch all messages (returns list directly)
    messages = get_all_messages()  # Now messages is the list directly
    
    if not messages:
        return None, None
    
    latest_message = max(messages, key=lambda msg: msg["id"])
    return latest_message["id"], latest_message["conversation_id"], latest_message["media_url"], latest_message["media_type"]



def get_conversation_messages(conversation_id):
    """
    Retrieves all messages that belong to a specific conversation.
    """
    messages = get_all_messages()  # Direct list of messages
    return [msg for msg in messages if msg.get("conversation_id") == conversation_id]



def send_message(conversation_id, role, content, media_url=None, media_type=None):
    """
    Sends a new message by inserting a row into the messages table in the online database.
    
    Parameters:
      conversation_id (int or str): The identifier for the conversation where the message belongs.
      role (str): The sender's role (e.g., "user", "assistant", "system").
      content (str): The textual content of the message (can be plain text or a JSON-formatted string).
      media_url (str, optional): URL or file path for any associated media (default is None).
      media_type (str, optional): The type of media (e.g., "image", "video"); default is None.
      
    Process:
      1. Call insert_message() with the provided parameters to add the message to the database.
      
    Returns:
      None. This function performs the insertion as a side effect.
    """
    # Insert the new message into the database.
    insert_message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        media_url=media_url,
        media_type=media_type
    )

# ------------------------------

def get_all_messages():
    SUPABASE_URL = "https://yjmsjtzfsggofmvraypd.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlqbXNqdHpmc2dnb2ZtdnJheXBkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODE2MzU4MywiZXhwIjoyMDUzNzM5NTgzfQ.281ZlrBrOS1AawSGkPFlDK22UqbDbp4yBGUJqHjIaJQ"
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    
    # Use desc=False for ascending order.
    response = supabase.table("messages").select("*").order("id", desc=False).execute()
    return response.data

def create_conversation(title):
    SUPABASE_URL = "https://yjmsjtzfsggofmvraypd.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlqbXNqdHpmc2dnb2ZtdnJheXBkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODE2MzU4MywiZXhwIjoyMDUzNzM5NTgzfQ.281ZlrBrOS1AawSGkPFlDK22UqbDbp4yBGUJqHjIaJQ"
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    response = supabase.table("conversations").insert({"title": title}).execute()
    return response.data

def insert_message(conversation_id, content, role, media_url=None, media_type=None):
    SUPABASE_URL = "https://yjmsjtzfsggofmvraypd.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlqbXNqdHpmc2dnb2ZtdnJheXBkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODE2MzU4MywiZXhwIjoyMDUzNzM5NTgzfQ.281ZlrBrOS1AawSGkPFlDK22UqbDbp4yBGUJqHjIaJQ"
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    message = {
        "conversation_id": int(conversation_id),
        "content": content,
        "role": role,
        "media_url": media_url,
        "media_type": media_type,
    }
    response = supabase.table("messages").insert(message).execute()
    return response.data
