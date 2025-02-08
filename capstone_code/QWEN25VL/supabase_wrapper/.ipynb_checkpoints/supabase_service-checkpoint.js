#!/usr/bin/env node
/**
 * supabase_service.js
 * 
 * This Node.js script wraps Supabase operations using your keys.
 * It supports the following commands:
 *   - get_all_messages
 *   - create_conversation <title>
 *   - insert_message <conversation_id> <content> <role> [media_url] [media_type]
 *
 * Note: When passing optional arguments for media_url and media_type, if you want to store null,
 * simply pass the string "None".
 */

const { createClient } = require('@supabase/supabase-js');

// Use your provided keys:
const SUPABASE_URL = "https://yjmsjtzfsggofmvraypd.supabase.co";
const SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlqbXNqdHpmc2dnb2ZtdnJheXBkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODE2MzU4MywiZXhwIjoyMDUzNzM5NTgzfQ.281ZlrBrOS1AawSGkPFlDK22UqbDbp4yBGUJqHjIaJQ";

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

async function getAllMessages() {
  try {
    const { data, error } = await supabase
      .from('messages')
      .select('*')
      .order('id', { ascending: true });
    if (error) throw error;
    console.log(JSON.stringify({ messages: data }));
  } catch (err) {
    console.error(JSON.stringify({ error: err.message || err }));
    process.exit(1);
  }
}

async function createConversation(title) {
  try {
    const { data, error } = await supabase
      .from('conversations')
      .insert([{ title }])
      .select();
    if (error) throw error;
    console.log(JSON.stringify({ conversation: data }));
  } catch (err) {
    console.error(JSON.stringify({ error: err.message || err }));
    process.exit(1);
  }
}

/**
 * Inserts a new message into the "messages" table.
 * Parameters:
 *   conversation_id: (number or string) the conversation ID  
 *   content: (string) message content  
 *   role: (string) e.g. "assistant" or "user"  
 *   media_url: (string) if "None" then it is set to null  
 *   media_type: (string) if "None" then it is set to null  
 */
async function insertMessage(conversation_id, content, role, media_url, media_type) {
  try {
    const { data, error } = await supabase
      .from('messages')
      .insert([
        {
          conversation_id: Number(conversation_id),
          content: content,
          role: role,
          media_url: media_url === "None" ? null : media_url,
          media_type: media_type === "None" ? null : media_type,
        }
      ])
      .select();
    if (error) throw error;
    console.log(JSON.stringify({ message: data }));
  } catch (err) {
    console.error(JSON.stringify({ error: err.message || err }));
    process.exit(1);
  }
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length < 1) {
    console.error("Usage: node supabase_service.js <command> [args]");
    process.exit(1);
  }
  const command = args[0];
  if (command === "get_all_messages") {
    await getAllMessages();
  } else if (command === "create_conversation") {
    if (args.length < 2) {
      console.error("Usage: node supabase_service.js create_conversation <title>");
      process.exit(1);
    }
    const title = args.slice(1).join(" ");
    await createConversation(title);
  } else if (command === "insert_message") {
    if (args.length < 4) {
      console.error("Usage: node supabase_service.js insert_message <conversation_id> <content> <role> [media_url] [media_type]");
      process.exit(1);
    }
    const conversation_id = args[1];
    const content = args[2];
    const role = args[3];
    // Optional arguments: if not provided, default to "None"
    const media_url = args[4] || "None";
    const media_type = args[5] || "None";
    await insertMessage(conversation_id, content, role, media_url, media_type);
  } else {
    console.error("Unknown command. Use 'get_all_messages', 'create_conversation', or 'insert_message'.");
    process.exit(1);
  }
}

main();
