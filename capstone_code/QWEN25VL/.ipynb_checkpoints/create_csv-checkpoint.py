#!/usr/bin/env python3
"""
create_csv.py

This script uses the Python wrapper (which calls your Node.js service) to fetch all messages
from your online Supabase database and then writes them to a CSV file called "messages_table.csv"
on your HPC.
"""

import csv
import json
from supabase_wrapper import get_all_messages

def main():
    try:
        # Fetch messages from the online database.
        # This returns a dictionary with a "messages" key.
        result = get_all_messages()
        messages = result.get("messages", [])
    except Exception as e:
        print("Error retrieving messages:", e)
        return

    if not messages:
        print("No messages found in the database.")
        return

    # Determine the CSV column headers from the first message.
    # You can adjust the order or columns if needed.
    fieldnames = list(messages[0].keys())

    # Write the messages to messages_table.csv
    try:
        with open("messages_table.csv", mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for message in messages:
                writer.writerow(message)
        print(f"CSV file 'messages_table.csv' created successfully with {len(messages)} records.")
    except Exception as e:
        print("Error writing CSV file:", e)

if __name__ == "__main__":
    main()
