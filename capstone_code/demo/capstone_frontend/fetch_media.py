import paramiko
from scp import SCPClient
import sys
import os

# HPC Config
HOST = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
USERNAME = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
PASSWORD = "REPLACE THIS WITH YOUR CHOSEN STORAGE MECHANISM"
# This is the root folder, but you'll likely get the full path from
# the argument "remote_file" (which might already have /scratch/whatever).
# So you won't always need REMOTE_DIR if your HPC path is absolute.

def create_ssh_client(host, username, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)
    return client

if __name__ == "__main__":
    """
    Usage: python fetch_media.py <remote_file_full_path> <local_temp_file>
      e.g. python fetch_media.py /scratch/.../images/some_image.jpg /tmp/some_image.jpg

    This script scp-GETs the HPC file to the local_temp_file,
    then prints that local_temp_file path to stdout on success.
    Exits with code 0 on success, 1 on failure.
    """
    if len(sys.argv) < 3:
        print("Error: Not enough arguments. Provide remote_path & local_path.")
        sys.exit(2)

    remote_file = sys.argv[1]
    local_file = sys.argv[2]

    try:
        ssh = create_ssh_client(HOST, USERNAME, PASSWORD)
        scp = SCPClient(ssh.get_transport())

        # Download the file from HPC
        scp.get(remote_file, local_file)

        # Print the local path so the Node side can read it
        print(local_file)

        scp.close()
        ssh.close()
        sys.exit(0)
    except Exception as e:
        print(f"Error fetching media: {e}")
        sys.exit(1)
