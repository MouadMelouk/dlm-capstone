import paramiko
from scp import SCPClient
import os
import sys
import time

# HPC config
HOST = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
USERNAME = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
PASSWORD = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
REMOTE_DIR = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"

def create_ssh_client(host, username, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)
    return client

if __name__ == "__main__":
    """
    Usage:
      python upload_file.py <local_file_path> <relative_remote_path>

    Example:
      python upload_file.py /tmp/abcd-1234.mp4 videos/abcd-1234.mp4
    This will upload /tmp/abcd-1234.mp4 to 
      /scratch/mmm9912/Capstone/FRONT_END_STORAGE/videos/abcd-1234.mp4
    and then print the remote path.
    """

    if len(sys.argv) < 3:
        print("Error: Not enough arguments.\nUsage: python upload_file.py <local_file_path> <relative_remote_path>")
        sys.exit(1)

    local_file = sys.argv[1]
    relative_remote_path = sys.argv[2].lstrip('/')  # ensure no leading slash

    # Construct final HPC path
    # e.g. /scratch/mmm9912/Capstone/FRONT_END_STORAGE/videos/<uuid>.mp4
    final_remote_path = os.path.join(REMOTE_DIR, relative_remote_path)

    try:
        ssh = create_ssh_client(HOST, USERNAME, PASSWORD)
        scp = SCPClient(ssh.get_transport())

        # Make sure the remote subdirectory exists:
        # Paramiko doesn't have a direct "mkdir -p" approach, so we can do an SSH exec:
        remote_dir = os.path.dirname(final_remote_path)
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p '{remote_dir}'")
        _ = stdout.read()  # not used
        _ = stderr.read()  # not used

        # Now upload
        scp.put(local_file, final_remote_path)

        # Print the final HPC path so our Node server can parse it
        print(final_remote_path)

        scp.close()
        ssh.close()
        sys.exit(0)  # success

    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)
