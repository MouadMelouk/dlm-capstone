import asyncio
import signal
import sys

from realtime import AsyncRealtimeClient

# ---------------------------
# 1) HARD-CODED SUPABASE DATA
# ---------------------------
SUPABASE_URL = "https://yjmsjtzfsggofmvraypd.supabase.co"
# Using service role key so you don't get a 401. 
SUPABASE_SERVICE_ROLE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlqbXNqdHpmc2dnb2ZtdnJheXBkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODE2MzU4MywiZXhwIjoyMDUzNzM5NTgzfQ.281ZlrBrOS1AawSGkPFlDK22UqbDbp4yBGUJqHjIaJQ"
)

# Derive the project ref. For instance, "yjmsjtzfsggofmvraypd"
project_ref = SUPABASE_URL.split("//")[1].split(".")[0]

# Construct the Realtime WebSocket URL
REALTIME_URL = f"wss://{project_ref}.supabase.co/realtime/v1/websocket"


# --------------------------------------
# 2) CALLBACK FOR NEW INSERTS ON 'messages'
# --------------------------------------
def on_insert(payload):
    """Called when a new row is inserted into the 'messages' table."""
    new_row = payload.get("new", {})
    role = new_row.get("role", "unknown")
    conversation_id = new_row.get("conversation_id", "unknown")
    content = new_row.get("content", "")
    media_type = new_row.get("media_type", "")
    media_url = new_row.get("media_url", "")

    print("\n--- New Message Inserted ---")
    print(f"From: {role}")
    print(f"Conversation: {conversation_id}")
    print(f"Content: {content}")
    print(f"Media Type: {media_type}")
    print(f"Media URL: {media_url}")
    print("--- End Message ---\n")


# We'll set a global stop flag when we catch Ctrl+C
stop_flag = False


def handle_sigint(signum, frame):
    """Called when Ctrl+C is pressed."""
    global stop_flag
    stop_flag = True  # Signal the main async loop to shut down.


# ---------------------------------------
# 3) MAIN ASYNC FUNCTION
# ---------------------------------------
async def main():
    # Create the Realtime client and connect.
    client = AsyncRealtimeClient(REALTIME_URL, SUPABASE_SERVICE_ROLE_KEY)
    await client.connect()

    # Create a channel for the 'messages' table (public schema).
    channel = client.channel("db-changes")
    await channel.subscribe()
    channel.on_postgres_changes("INSERT", schema="public", table="messages", callback=on_insert)
    
    print("Listening for INSERT events on 'public.messages' table.")
    print("Press Ctrl+C once to stop.\n")

    # Instead of blocking on client.listen(), we run it as a task
    # so Ctrl+C can be handled gracefully.
    listen_task = asyncio.create_task(client.listen())

    # We'll loop here, checking if the user pressed Ctrl+C
    while not stop_flag:
        await asyncio.sleep(0.2)

    # Once stop_flag is True, we cancel the background listening task.
    listen_task.cancel()
    try:
        await listen_task
    except asyncio.CancelledError:
        pass

    # Now close the client.
    await client.close()
    print("\nShut down gracefully. Bye!")


# ---------------------------------------
# 4) ENTRY POINT
# ---------------------------------------
if __name__ == "__main__":
    # Register our signal handler for Ctrl+C
    signal.signal(signal.SIGINT, handle_sigint)

    # Run the main coroutine until it finishes.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # If for some reason the user hits Ctrl+C again while shutting down:
        print("\nForce exit.")
        sys.exit(1)
