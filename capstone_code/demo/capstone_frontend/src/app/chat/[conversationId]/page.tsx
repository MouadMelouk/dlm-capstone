'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';

import 'katex/dist/katex.min.css';       // KaTeX styles
import 'highlight.js/styles/github.css'; // Or another highlight.js theme

import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { createClient } from '@supabase/supabase-js';
import { PaperAirplaneIcon, PaperClipIcon, XMarkIcon, HomeIcon } from '@heroicons/react/24/outline';
import { Loader2 } from 'lucide-react';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

// Helper to guess media type by file extension (very simplistic)
function guessMediaType(url: string): 'image' | 'video' | undefined {
  const lowerUrl = url.toLowerCase();
  if (
    lowerUrl.endsWith('.jpg') ||
    lowerUrl.endsWith('.jpeg') ||
    lowerUrl.endsWith('.png') ||
    lowerUrl.endsWith('.gif')
  ) {
    return 'image';
  }
  if (
    lowerUrl.endsWith('.mp4') ||
    lowerUrl.endsWith('.mov') ||
    lowerUrl.endsWith('.avi')
  ) {
    return 'video';
  }
  return undefined;
}

// <-- NEW: Transformation function for math delimiters -->
function transformMathDelimiters(text: string): string {
  return text
    // Convert display math: \[ ... \]  →  $$ ... $$
    .replace(/\\\[/g, '$$')
    .replace(/\\\]/g, '$$')
    // Convert inline math: \( ... \)  →  $ ... $
    .replace(/\\\(/g, '$')
    .replace(/\\\)/g, '$');
}

export default function ChatPage() {
  const { conversationId } = useParams();
  const [message, setMessage] = useState('');
  const [mediaUrl, setMediaUrl] = useState('');
  const [chat, setChat] = useState<
    { type: 'sent' | 'received'; text: string; mediaUrl?: string; mediaType?: string }[]
  >([]);
  const [loadingMessages, setLoadingMessages] = useState(true);
  const [sending, setSending] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [enlargedImage, setEnlargedImage] = useState<string | null>(null);

  // Helper to render media content (image or video)
  const renderMedia = (msg: { mediaUrl?: string; mediaType?: string }) => {
    if (!msg.mediaUrl) return null;
  
    const mediaUrl = msg.mediaUrl;
  
    if (msg.mediaType === 'image') {
      return (
        <img
          src={`/api/fetchMedia?path=${encodeURIComponent(mediaUrl)}`}
          alt="Media"
          className="rounded-lg max-w-full h-auto cursor-pointer"
          onClick={() =>
            setEnlargedImage(`/api/fetchMedia?path=${encodeURIComponent(mediaUrl)}`)
          }
        />
      );
    }
    if (msg.mediaType === 'video') {
      return (
        <video
          src={`/api/fetchMedia?path=${encodeURIComponent(mediaUrl)}`}
          controls
          className="rounded-lg max-w-full h-auto"
        />
      );
    }
    return null;
  };

  useEffect(() => {
    if (!conversationId) return;

    const fetchMessages = async () => {
      setLoadingMessages(true);
      const { data, error } = await supabase
        .from('messages')
        .select('*')
        .eq('conversation_id', conversationId)
        .order('id', { ascending: true });

      if (!error && data) {
        setChat(
          data.map((msg: any) => ({
            type: msg.role === 'user' ? 'sent' : 'received',
            text: msg.content,
            mediaUrl: msg.media_url,
            mediaType: msg.media_type,
          }))
        );
      }
      setLoadingMessages(false);
    };

    fetchMessages();

    const subscription = supabase
      .channel('conversation')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'messages',
          filter: `conversation_id=eq.${Number(conversationId)}`,
        },
        (payload) => {
          setChat((prevChat) => [
            ...prevChat,
            {
              type: payload.new.role === 'user' ? 'sent' : 'received',
              text: payload.new.content,
              mediaUrl: payload.new.media_url,
              mediaType: payload.new.media_type,
            },
          ]);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(subscription);
    };
  }, [conversationId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chat]);

  const handleSend = async () => {
    if (!conversationId || (!message.trim() && !mediaUrl.trim()) || sending) return;

    setSending(true);
    try {
      const type = mediaUrl ? guessMediaType(mediaUrl) : undefined;

      const response = await fetch('/api/sendMessage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          message,
          mediaUrl: mediaUrl || null,
          mediaType: type || null,
        }),
      });

      if (response.ok) {
        setMessage('');
        setMediaUrl('');
      }
    } catch (err) {
      console.error('Error sending message:', err);
    } finally {
      setSending(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const selectedFile = e.target.files[0];

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/api/uploadFile', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (data.hpcPath) {
        setMediaUrl(data.hpcPath);
      } else {
        console.error('Upload failed: ', data);
        alert('Upload failed. See console for details.');
      }
    } catch (err) {
      console.error('Error uploading file: ', err);
      alert('Error uploading file. See console for details.');
    }
  }, []);

  return (
    <div className="flex flex-col min-h-screen
                bg-gradient-to-br from-gray-900 via-[#1F1C2C] to-[#928DAB]
                text-white">
      {/* HEADER */}
      <header
        className="sticky top-0 z-10 flex items-center justify-center h-16 
                   bg-black/40 backdrop-blur-md shadow-md relative"
      >
        {/* Home Button */}
        <Link href="/" className="absolute left-4">
          <button
            className="flex items-center justify-center w-10 h-10 
                       bg-gray-500/50 hover:bg-gray-600/70 transition-colors 
                       rounded-full shadow-md relative group"
            aria-label="Go to Homepage"
          >
            <HomeIcon className="w-5 h-5 text-white" />
            {/* Tooltip */}
            <span className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 
                             bg-gray-800 text-white text-xs rounded py-1 px-3 
                             min-w-[160px] whitespace-nowrap
                             opacity-0 group-hover:opacity-100 transition-opacity 
                             pointer-events-none">
              Go to Homepage
            </span>
          </button>
        </Link>
        <p className="text-xl font-semibold">FakeFinder — Conversation {conversationId}</p>
      </header>

      {/* MAIN CHAT BODY */}
      <main className="flex-grow w-full flex justify-center overflow-y-auto">
        <div className="w-full md:w-1/2 flex flex-col gap-4 p-4">
          {loadingMessages ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="animate-spin w-5 h-5 mr-2" />
              <span>Loading your conversation history...</span>
            </div>
          ) : (
            (() => {
              const renderedMessages = [];
              for (let i = 0; i < chat.length; i++) {
                const message = chat[i];
                const isSent = message.type === 'sent';
                const bubbleClasses = isSent
  ? 'bg-blue-800 text-white backdrop-blur-md shadow-md'
  : 'bg-black/30 text-white backdrop-blur-md shadow-md';

                const alignment = isSent ? 'items-end' : 'items-start';

                // Check if this is a media-only message (no text)
                const isMediaOnly =
                  (!message.text || message.text.trim() === '') && message.mediaUrl;

                // And if the following message is also media-only from the same sender…
                if (
                  isMediaOnly &&
                  i + 1 < chat.length &&
                  (!chat[i + 1].text || chat[i + 1].text.trim() === '') &&
                  chat[i + 1].mediaUrl &&
                  chat[i + 1].type === message.type
                ) {
                  renderedMessages.push(
                    <div key={`group-${i}`} className={`flex flex-col ${alignment} gap-1`}>
                      <div
                        className={`inline-flex flex-row gap-2 p-3 rounded-xl text-sm shadow-md ${bubbleClasses}`}
                      >
                        {renderMedia(message)}
                        {renderMedia(chat[i + 1])}
                      </div>
                    </div>
                  );
                  i++; // Skip the next message since it's already grouped.
                  continue;
                }

                // Otherwise, render the message normally.
                renderedMessages.push(
                  <div key={i} className={`flex flex-col ${alignment} gap-1`}>
                    {message.text && (
                      <div
                      className={`inline-block w-full md:w-[90%] p-3 rounded-xl text-sm shadow-md ${bubbleClasses}`}
                      >
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex, rehypeHighlight]}
                        >
                          {transformMathDelimiters(message.text)}
                        </ReactMarkdown>
                      </div>
                    )}
                    {message.mediaUrl && (
                      <div
                        className={`inline-block max-w-[35%] p-3 rounded-xl text-sm shadow-md ${bubbleClasses}`}
                      >
                        {renderMedia(message)}
                      </div>
                    )}
                  </div>
                );
              }
              return renderedMessages;
            })()
          )}

          <div ref={chatEndRef} />
        </div>
      </main>

      {/* Enlarged Image Overlay */}
      {enlargedImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50"
          onClick={() => setEnlargedImage(null)}
        >
          <div
            className="relative"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="absolute top-2 right-2 bg-black bg-opacity-50 p-2 rounded-full text-white"
              onClick={() => setEnlargedImage(null)}
            >
              <XMarkIcon className="w-6 h-6" />
            </button>
            <img src={enlargedImage} alt="Enlarged Media" className="max-w-full max-h-screen rounded-lg" />
          </div>
        </div>
      )}

      {/* FOOTER: INPUTS */}
      <footer
        className="sticky bottom-0 z-10 w-full h-16 flex items-center justify-center
                   bg-black/40 backdrop-blur-md"
      >
        <div className="flex items-center gap-2 w-full md:w-1/2 px-4">
          <input
            type="file"
            accept="video/*"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />

          <div className="relative group">
            <button
              className="bg-black/50 hover:bg-black/70 p-3 rounded-full 
                         transition-colors duration-300 shadow-md"
              onClick={() => fileInputRef.current?.click()}
              aria-label="Upload File"
            >
              <PaperClipIcon className="w-5 h-5 text-white" />
            </button>
            <span className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 
                             bg-gray-800 text-white text-xs rounded py-1 px-3 
                             min-w-[160px] whitespace-nowrap
                             opacity-0 group-hover:opacity-100 transition-opacity 
                             pointer-events-none">
              Supported formats: mp4, mov, avi
            </span>
          </div>

          {mediaUrl && (
            <div className="bg-black/50 text-sm px-2 py-1 rounded-lg shadow-md">
              {"File uploaded!"}
            </div>
          )}

          <input
            className="flex-grow bg-black/30 text-white rounded-full px-4 py-2 outline-none
                       shadow-inner placeholder-gray-300
                       focus:ring-2 focus:ring-blue-400 focus:outline-none"
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask FakeFinder..."
          />

          <div className="relative group">
            <button
              disabled={sending}
              className="bg-gradient-to-r from-blue-600 to-blue-500 p-3 rounded-full 
                         transition-colors duration-300 shadow-md hover:opacity-90 disabled:opacity-50"
              onClick={handleSend}
              aria-label="Send Message"
            >
              {sending ? (
                <Loader2 className="animate-spin w-5 h-5" />
              ) : (
                <PaperAirplaneIcon className="w-5 h-5 text-white transform rotate-45" />
              )}
            </button>
            <span className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 
                             bg-gray-800 text-white text-xs rounded py-1 px-3 
                             min-w-[160px] whitespace-nowrap
                             opacity-0 group-hover:opacity-100 transition-opacity 
                             pointer-events-none">
              Send a message
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
