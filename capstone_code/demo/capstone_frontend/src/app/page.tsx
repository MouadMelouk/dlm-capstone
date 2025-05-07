'use client';

import { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';
import { useRouter } from 'next/navigation';
import { Loader2, Trash2 } from 'lucide-react';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

export default function Home() {
  const router = useRouter();
  const [conversations, setConversations] = useState<{ id: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(true);

  useEffect(() => {
    const fetchConversations = async () => {
      setLoadingConversations(true);
      const { data, error } = await supabase
        .from('conversations')
        .select('id')
        .order('id', { ascending: false });

      if (!error && data) {
        setConversations(data);
      }
      setLoadingConversations(false);
    };

    fetchConversations();
  }, []);

  const startNewChat = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/createConversation', {
        method: 'POST',
      });
      const data = await response.json();

      if (data.success) {
        router.push(`/chat/${data.conversation_id}`);
      }
    } catch (err) {
      console.error('Error creating conversation', err);
    }
    setLoading(false);
  };

  const deleteConversation = async (conversationId: number) => {
    try {
      // POST to your deleteConversation API
      const response = await fetch('/api/deleteConversation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: conversationId }),
      });

      const data = await response.json();
      if (data.success) {
        // Remove the conversation from local state so UI updates
        setConversations((prev) => prev.filter((c) => c.id !== conversationId));
      } else {
        console.error('Failed to delete conversation:', data.error);
      }
    } catch (err) {
      console.error('Error deleting conversation:', err);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen 
                    bg-gradient-to-br from-gray-900 via-[#1F1C2C] to-[#928DAB] text-white">
      <div className="backdrop-blur-md bg-black/40 p-8 rounded-2xl shadow-xl border border-white/10 max-w-md w-full text-center">
        <h1 className="text-4xl font-extrabold mb-6 tracking-tight text-white">
          I am FakeFinder.
        </h1>
        <div className="flex justify-center">
          <button
            onClick={startNewChat}
            className="relative flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-blue-500 px-6 py-3 text-lg font-semibold rounded-lg transition-all hover:from-blue-500 hover:to-blue-400 disabled:from-gray-700 disabled:to-gray-600"
            disabled={loading}
          >
            {loading ? <Loader2 className="animate-spin w-5 h-5 text-white" /> : 'Start New Chat'}
          </button>
        </div>

        {loadingConversations ? (
          <div className="mt-8 flex items-center justify-center">
            <Loader2 className="animate-spin w-5 h-5 mr-2 text-white" />
            <span>Loading your previous conversations...</span>
          </div>
        ) : (
          conversations.length > 0 && (
            <div className="mt-8">
              <p className="text-lg mb-4 text-gray-200">Previous Chats:</p>
              <ul className="space-y-3">
                {conversations.map((conv) => (
                  <li
                    key={conv.id}
                    className="group flex items-center justify-between transition-transform transform hover:scale-105"
                  >
                    <a
                      href={`/chat/${conv.id}`}
                      className="flex-1 px-4 py-3 bg-black/30 backdrop-blur-md rounded-lg shadow-md text-blue-300 hover:text-blue-200 transition-colors"
                    >
                      Conversation {conv.id}
                    </a>
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        const ok = window.confirm(
                          `Heads up! You're about to wipe conversation ${conv.id} from existence. This can’t be undone. Proceed?`
                        );

                        if (ok) {
                          deleteConversation(conv.id);
                        }
                      }}
                      className="ml-3 p-2 text-red-500 hover:text-red-400 transition-colors rounded-lg hover:bg-red-900/20"
                      aria-label="Delete conversation"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )
        )}
      </div>
    </div>
  );
}
