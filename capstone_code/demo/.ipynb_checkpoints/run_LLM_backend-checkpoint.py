import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import time
import json
from utils.prompting_utils import *
from utils.system_prompt import messages as system_prompt

torch._dynamo.config.suppress_errors = True

# Set all cache directories explicitly

# REPLACE THIS
root_directory = "/scratch/mmm9912/"

os.environ["HF_HOME"] = root_directory + "cache"
os.environ["TORCH_HOME"] = root_directory + "cache/torch"
os.environ["TFHUB_CACHE_DIR"] = root_directory + "cache/tensorflow"
os.environ["XDG_CACHE_HOME"] = root_directory + "cache"
os.environ["HF_DATASETS_CACHE"] = root_directory + "cache/huggingface_datasets"
os.environ["PIP_CACHE_DIR"] = root_directory + "cache/pip"

cache_dir = cache_dir + "cache"

# Set C++ compiler from sh
print("CC set to:", os.environ["CC"])
print("CXX set to:", os.environ["CXX"])


# Initialize the tokenizer
chosen_LLM = "Qwen/Qwen2.5-32B-Instruct" # choose best variant for target hardware REPLACE THIS
tokenizer = AutoTokenizer.from_pretrained(chosen_LLM, cache_dir=cache_dir)

# Pass the default decoding hyperparameters of Qwen2.5-32B-Instruct
# max_tokens is for the maximum length for generation (so: NEW tokens).
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=4096)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=chose_LLM, download_dir=cache_dir, tensor_parallel_size=2)


import time

# Initialize the last processed message ID to track only new messages
last_processed_id, _ , _ , _ = get_latest_message_info()
print(f"Last message ID in the database: {last_processed_id}")

established_media_url = None
established_media_type = None

while True:
    # Step 1: Surveil for new messages
    latest_id, conv_id, media_url, media_type = get_latest_message_info()

    if isinstance(media_type, str) and "video" in media_type:
        established_media_url = media_url
        established_media_type = media_type
    
    if latest_id <= last_processed_id:
        last_processed_id = latest_id
        print(f"Last message ID: {last_processed_id}, waiting for new message...")
        time.sleep(1)
        continue
    
    if latest_id > last_processed_id:
        print(f"Found new message {latest_id} in conversation {conv_id}!")
        last_processed_id = latest_id
        
        # Get conversation messages
        sorted_messages = sorted(get_conversation_messages(conv_id),  # Changed here
                                key=lambda msg: msg["id"])
        
        # Build the messages list for the tokenizer
        messages = system_prompt.copy()
        for msg in sorted_messages:
            messages.append(format_user_or_assistant_message(msg["role"], msg["content"] + f"Uploaded media: {established_media_type}" + f", path: {established_media_url}"))
        
        counter = 0
        a_video_was_processed = False
        while True:
            print(f"Trial #{counter+1}...")
            # Step 3: Generate initial response from LLM
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = llm.generate([text], sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                break
        
            print(f"\n\nGenerated text: {generated_text!r}\n\n")
        
            # Step 4: Process the generated text
            try:
                handle_result = handle_llm_response(generated_text)
                print("Processing succeeded.")
                a_video_was_processed = True
                break  # Exit the loop if no error occurs
            except Exception as e:
                counter += 1
                if counter >= 5:
                    send_message(
                        conversation_id=conv_id,
                        role="assistant",
                        content=("The video format you uploaded seems to be corrupted or non-standard. In the meantime, do you have another video to analyze?"),
                        media_url=None,
                        media_type=None
                    )
                    latest_id += 1
                    a_video_was_processed = False
                    break
                print(f"An error occurred: {e}. Retrying...")
                time.sleep(1)

        if a_video_was_processed:
            if isinstance(handle_result, str):
                assistant_response = handle_result
                gradcam_paths = []
            else:
                # Step 5: Handle expert feedback and GradCAM paths
                expert_feedback = handle_result[0] 
                gradcam_paths = handle_result[1:]
                
                # Append expert feedback as a system message
                expert_response = format_expert_message(expert_feedback)
                messages.append(expert_response)
                
                # Step 6: Generate response with expert context
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                outputs = llm.generate([text], sampling_params)
                for output in outputs:
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    break
                
                # Step 7: Process the final response
                assistant_response = handle_llm_response(generated_text)
    
            if "Uploaded media:" in assistant_response:
                assistant_response = assistant_response.split("Uploaded media:")[0].strip()
            
            # Step 8.1: Send the assistant's response
            send_message(
                conversation_id=conv_id,
                role="assistant",
                content=(assistant_response),
                media_url=None,
                media_type=None
            )
            latest_id += 1
            
            # Step 8.2: Send GradCAM images as media messages
            for idx, path in enumerate(gradcam_paths):
                send_message(
                    conversation_id=conv_id,
                    role="assistant",
                    content="",
                    media_url=path,
                    media_type="image"
                )
                latest_id += 1
        last_processed_id = latest_id
    # Polling interval
    time.sleep(1)
