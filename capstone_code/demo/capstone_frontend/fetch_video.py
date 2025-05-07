import paramiko
from scp import SCPClient
import sys
import os

# HPC Config (uses ssh private key)
HOST = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
USERNAME = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
PASSWORD = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
REMOTE_DIR = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM/FRONT_END_STORAGE"

def create_ssh_client(host, username, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)
    return client

if __name__ == "__main__":
    """
    Usage: python fetch_video.py <remote_file_full_path> <local_temp_file>
      e.g. python fetch_video.py /scratch/.../abcd-1234.mp4 /tmp/abcd-1234.mp4

    We will scp GET the remote file into local_temp_file,
    then print the local_temp_file path to stdout on success.
    """
    if len(sys.argv) < 3:
        print("Error: Not enough arguments.")
        sys.exit(2)

    remote_file = sys.argv[1]  # e.g. /scratch/mmm9912/Capstone/FRONT_END_STORAGE/videos/abcd-1234.mp4
    local_file = sys.argv[2]   # e.g. /tmp/abcd-1234.mp4

    try:
        ssh = create_ssh_client(HOST, USERNAME, PASSWORD)
        scp = SCPClient(ssh.get_transport())

        # Download the file
        scp.get(remote_file, local_file)

        # Print the local path so Node.js can read it
        print(local_file)

        scp.close()
        ssh.close()
        sys.exit(0)
    except Exception as e:
        print(f"Error fetching video: {e}")
        sys.exit(1)
