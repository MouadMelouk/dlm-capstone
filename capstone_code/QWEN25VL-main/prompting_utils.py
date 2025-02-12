import cv2
import os
import uuid
import json

import os
from supabase import create_client, Client

def extract_k_frames(video_path, k):
    """
    Extracts k equally spaced frames from a video and returns a list of tuples.
    Each tuple contains:
       (original_frame_index, saved_image_file_path)
    """
    save_dir = "/scratch/mmm9912/Capstone/FRONT_END_STORAGE/images/"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < k:
        cap.release()
        raise ValueError(f"Video has only {frame_count} frames, but {k} are required.")

    # Compute k equally spaced frame indices.
    frame_indices = [int(i * frame_count / k) for i in range(k)]
    saved_frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Failed to read frame at index {idx}")

        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(save_dir, filename)
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


def consult_expert_model(video_path, k, model_name, cuda, manual_seed):
    """
    Extracts frames from the video, runs inference on each extracted image,
    and returns a list of strings where:
    
      - The first string is a summary description of the analysis.
      - The subsequent strings are the overlay image paths for visualization.
    
    The summary string is formatted as:
    
      "The frames analyzed by {MODEL} were {frame_indices}. 
       In frame {idx1} forgery was detected with confidence {conf1}%, 
       in frame {idx2} forgery was detected with confidence {conf2}%, 
       ... The average confidence was {avg_confidence}%. 
       The areas that potentially contain adversarial artifacts comprise of 
       {red1}, {red2}, ... of each frame, respectively."
    """
    # Extract frames: list of (frame_index, image_path)
    frames = extract_k_frames(video_path, k)
    image_paths = [path for (_, path) in frames]
    
    # Run inference on the extracted images.
    # Each element returned is assumed to be:
    # (overlay_path, confidence, prediction_message, red_percentage)
    inference_results = wrapped_run_inference_on_images_with_old_preprocess(model_name, image_paths, cuda, manual_seed)
    
    # Combine each frame's info with its inference result.
    combined_results = []
    for (frame_index, _), (overlay_path, confidence, prediction_message, red_percentage) in zip(frames, inference_results):
        conf_percent = confidence * 100  # Convert to percentage
        combined_results.append({
            "frame_index": frame_index,
            "overlay_path": overlay_path,
            "confidence": conf_percent,
            "message": prediction_message,
            "red_percentage": red_percentage if "detected forgery" in prediction_message.lower() else None
        })

    print(combined_results)
    # Build the summary string.
    # List of frame indices.
    frame_indices_str = ", ".join(str(item["frame_index"]) for item in combined_results)
    
    # Build per-frame details.
    details = []
    red_values = []
    total_confidence = 0.0
    for item in combined_results:
        total_confidence += item["confidence"]
        # If forgery was detected, use a standard phrasing.
        if item["red_percentage"] is not None:
            details.append(f"In frame {item['frame_index']} forgery was detected with confidence {item['confidence']:.1f}%")
            red_values.append(f"{item['red_percentage']:.1f}%")
        else:
            details.append(f"In frame {item['frame_index']} no forgery was detected with confidence {item['confidence']:.1f}%")
    
    details_str = ", ".join(details)
    avg_confidence = total_confidence / len(combined_results) if combined_results else 0.0
    
    # Build the red-highlight string (only if there are any red values).
    red_str = ""
    if red_values:
        red_str = f" The area of the image that potentially contains adversarial artifacts comprise of {', '.join(red_values)} of each frame, respectively."
    
    summary = (f"The assistant has made an expert function call to {model_name.upper()}. The frames analyzed by {model_name.upper()} were {frame_indices_str}. "
               f"{details_str}. The average confidence was {avg_confidence:.1f}%." + red_str)
    
    # The final output: first element is the summary; the rest are overlay paths.
    visualization_paths = [item["overlay_path"] for item in combined_results]
    return [summary] + visualization_paths


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
