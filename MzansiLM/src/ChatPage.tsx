import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatMessage {
  role: string;
  content: string;
}

export default function ChatPage() {
  const [modelPath, setModelPath] = useState("./models/bitnet-b1.58-2b-4t-bf16");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Accumulate streamed tokens without stale-closure issues
  const streamRef = useRef("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Register Tauri event listeners once on mount
  useEffect(() => {
    const unsubs: UnlistenFn[] = [];

    listen<{ text: string }>("inference://token", (e) => {
      streamRef.current += e.payload.text;
      setStreamingText(streamRef.current);
    }).then((u) => unsubs.push(u));

    listen<{ success: boolean; error?: string }>("inference://done", (e) => {
      setIsGenerating(false);
      if (e.payload.success) {
        const finalText = streamRef.current;
        if (finalText.trim()) {
          setMessages((prev) => [...prev, { role: "assistant", content: finalText }]);
        }
      } else {
        setError(e.payload.error ?? "Inference failed");
      }
      streamRef.current = "";
      setStreamingText("");
    }).then((u) => unsubs.push(u));

    return () => unsubs.forEach((u) => u());
  }, []);

  // Scroll to bottom whenever messages or streaming text changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  async function send() {
    const text = input.trim();
    if (!text || isGenerating) return;

    setInput("");
    setError(null);
    streamRef.current = "";
    setStreamingText("");

    // Build full message list for the backend
    const allMessages: ChatMessage[] = [];
    if (systemPrompt.trim()) {
      allMessages.push({ role: "system", content: systemPrompt.trim() });
    }
    for (const m of messages) {
      allMessages.push({ role: m.role, content: m.content });
    }
    allMessages.push({ role: "user", content: text });

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setIsGenerating(true);

    try {
      await invoke("start_inference", {
        config: {
          model_path: modelPath,
          messages: allMessages,
          max_new_tokens: 512,
          temperature: 0.7,
        },
      });
    } catch (err) {
      setIsGenerating(false);
      setError(String(err));
      streamRef.current = "";
      setStreamingText("");
    }
  }

  async function stop() {
    try {
      await invoke("stop_inference");
    } catch {
      // ignore
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="chat-page">
      {/* Config bar */}
      <div className="chat-config">
        <div className="chat-config-field">
          <span className="chat-config-label">Model path</span>
          <input
            className="chat-config-input"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            disabled={isGenerating}
            placeholder="./models/bitnet-b1.58-2b-4t-bf16"
          />
        </div>
        <div className="chat-config-field chat-config-field--sys">
          <span className="chat-config-label">System</span>
          <input
            className="chat-config-input"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            disabled={isGenerating}
            placeholder="Optional system prompt…"
          />
        </div>
      </div>

      {/* Message area */}
      <div className="chat-messages">
        {messages.length === 0 && !streamingText && !isGenerating && (
          <div className="chat-empty">
            Enter a model path above and start chatting.
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`bubble bubble--${msg.role}`}>
            <span className="bubble-role">{msg.role === "user" ? "You" : "Assistant"}</span>
            <p className="bubble-text">{msg.content}</p>
          </div>
        ))}

        {isGenerating && (
          <div className="bubble bubble--assistant bubble--streaming">
            <span className="bubble-role">Assistant</span>
            <p className="bubble-text">
              {streamingText || <span className="chat-thinking">thinking…</span>}
            </p>
          </div>
        )}

        {error && <p className="error chat-error">{error}</p>}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div className="chat-input-row">
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message… (Enter to send, Shift+Enter for newline)"
          disabled={isGenerating}
          rows={1}
        />
        {isGenerating ? (
          <button className="btn-stop chat-send-btn" onClick={stop}>
            Stop
          </button>
        ) : (
          <button
            className="btn-start chat-send-btn"
            onClick={send}
            disabled={!input.trim()}
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}
