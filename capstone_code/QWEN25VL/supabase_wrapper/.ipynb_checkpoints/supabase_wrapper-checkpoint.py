#!/usr/bin/env python3
"""
supabase_wrapper.py

A Python wrapper that calls the Node.js supabase_service.js CLI tool.
It now provides three functions:
  - get_all_messages()  -> returns all messages from all conversations.
  - create_conversation(title)  -> creates a new conversation with the given title.
  - insert_message(conversation_id, content, role, media_url=None, media_type=None)
     -> inserts a new message row in the messages table.
"""

import subprocess
import json
import os

# Path to the Node.js CLI script
NODE_SCRIPT = os.path.join(os.path.dirname(__file__), "supabase_service.js")

def run_node_command(args):
    """
    Run the Node.js script with the provided arguments.
    Returns the parsed JSON output.
    """
    try:
        result = subprocess.run(
            ["node", NODE_SCRIPT] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        output = result.stdout.strip()
        return json.loads(output)
    except subprocess.CalledProcessError as e:
        print("Error running Node.js script:", e.stderr)
        raise
    except json.JSONDecodeError as e:
        print("Error decoding JSON output:", output)
        raise

def get_all_messages():
    """Get all messages from all conversations."""
    return run_node_command(["get_all_messages"])

def create_conversation(title):
    """Create a new conversation with the specified title."""
    return run_node_command(["create_conversation", title])

def insert_message(conversation_id, content, role, media_url=None, media_type=None):
    """
    Insert a new message row in the messages table.
    Parameters:
      conversation_id: (int or str)
      content: (str)
      role: (str) e.g. "assistant" or "user"
      media_url: (str or None) pass None to store null
      media_type: (str or None) pass None to store null
    """
    args = [
        "insert_message",
        str(conversation_id),
        content,
        role,
        media_url if media_url is not None else "None",
        media_type if media_type is not None else "None",
    ]
    return run_node_command(args)

if __name__ == "__main__":
    # Example usage:
    print("=== Inserting a new message ===")
    try:
        # Insert a message with:
        # conversation_id = 1, content = "Hey! Lolol", role = "assistant", media_url = None, media_type = None
        result = insert_message(1, "Hey! Lolol", "assistant")
        print("Inserted message:", json.dumps(result, indent=2))
    except Exception as e:
        print("Failed to insert message:", e)
