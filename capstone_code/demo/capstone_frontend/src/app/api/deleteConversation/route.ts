import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

export async function POST(req: Request) {
  try {
    const { conversation_id } = await req.json();

    if (!conversation_id) {
      return NextResponse.json(
        { error: 'conversation_id is required' },
        { status: 400 }
      );
    }

    // 1) Delete messages for this conversation
    const { error: deleteMessagesError } = await supabase
      .from('messages')
      .delete()
      .eq('conversation_id', conversation_id);

    if (deleteMessagesError) {
      return NextResponse.json(
        { error: deleteMessagesError.message },
        { status: 500 }
      );
    }

    // 2) Delete the conversation itself
    const { error: deleteConversationError } = await supabase
      .from('conversations')
      .delete()
      .eq('id', conversation_id);

    if (deleteConversationError) {
      return NextResponse.json(
        { error: deleteConversationError.message },
        { status: 500 }
      );
    }

    // Everything succeeded
    return NextResponse.json({ success: true }, { status: 200 });
  } catch (err) {
    return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
  }
}
