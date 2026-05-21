import { useEffect, useRef, useState } from "react";
import type { JsonlEntry, Message } from "../types";

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

interface JsonlEntryProps {
  entry: JsonlEntry;
  index: number;
  isTraining: boolean;
  onChange: (updated: JsonlEntry) => void;
  onDelete: (id: string) => void;
}

export default function JsonlEntryCard({
  entry,
  index,
  isTraining,
  onChange,
  onDelete,
}: JsonlEntryProps) {
  const [expanded, setExpanded] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  // Auto-expand + scroll into view when this card becomes the active training example
  useEffect(() => {
    if (isTraining) {
      setExpanded(true);
      cardRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [isTraining]);

  // Unique roles present in this entry, in order of first appearance
  const roleOrder = ["system", "user", "assistant"] as const;
  const presentRoles = roleOrder.filter((r) =>
    entry.messages.some((m) => m.role === r)
  );

  function updateMessage(msgIndex: number, key: keyof Message, value: string) {
    const updated = entry.messages.map((m, i) =>
      i === msgIndex ? { ...m, [key]: value } : m
    );
    onChange({ ...entry, messages: updated });
  }

  function deleteMessage(msgIndex: number) {
    onChange({ ...entry, messages: entry.messages.filter((_, i) => i !== msgIndex) });
  }

  function addMessage() {
    const lastRole = entry.messages[entry.messages.length - 1]?.role ?? "user";
    const nextRole: Message["role"] =
      lastRole === "system" ? "user" : lastRole === "user" ? "assistant" : "user";
    onChange({
      ...entry,
      messages: [...entry.messages, { role: nextRole, content: "" }],
    });
  }

  return (
    <div
      ref={cardRef}
      className={`entry-card${isTraining ? " training" : ""}`}
    >
      {/* Header */}
      <div
        className="entry-header"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="entry-index">#{index + 1}</span>

        <div className="entry-roles">
          {entry.parseError ? (
            <span className="role-pill" style={{ background: "#3a1a1a", color: "#f87171" }}>
              parse error
            </span>
          ) : (
            presentRoles.map((r) => (
              <span key={r} className={`role-pill ${r}`}>{r}</span>
            ))
          )}
        </div>

        {isTraining && <span className="training-badge">● TRAINING</span>}

        <span className={`entry-chevron${expanded ? " open" : ""}`}>▼</span>

        <button
          className="btn-sm btn-danger entry-delete"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(entry.id);
          }}
          title="Delete example"
        >
          ×
        </button>
      </div>

      {/* Body */}
      {expanded && (
        <div className="entry-body">
          {entry.parseError ? (
            <div className="entry-parse-error">
              <span className="error-label">Invalid JSON</span>
              <pre className="error-raw">{entry.raw}</pre>
              <span className="error-msg">{entry.parseError}</span>
            </div>
          ) : (
            <>
              {entry.messages.map((msg, i) => (
                <MessageRow
                  key={i}
                  message={msg}
                  onChangeRole={(v) => updateMessage(i, "role", v)}
                  onChangeContent={(v) => updateMessage(i, "content", v)}
                  onDelete={() => deleteMessage(i)}
                />
              ))}
              <button className="btn-sm btn-ghost add-msg-btn" onClick={addMessage}>
                + message
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MessageRow — one message inside an entry card
// ---------------------------------------------------------------------------

interface MessageRowProps {
  message: Message;
  onChangeRole: (role: string) => void;
  onChangeContent: (content: string) => void;
  onDelete: () => void;
}

function MessageRow({ message, onChangeRole, onChangeContent, onDelete }: MessageRowProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = el.scrollHeight + "px";
  }

  useEffect(() => {
    autoResize();
  }, [message.content]);

  return (
    <div className="msg-row">
      <select
        className="msg-role-select"
        value={message.role}
        onChange={(e) => onChangeRole(e.target.value)}
      >
        <option value="system">system</option>
        <option value="user">user</option>
        <option value="assistant">assistant</option>
      </select>
      <textarea
        ref={textareaRef}
        className="msg-content"
        value={message.content}
        rows={1}
        onChange={(e) => {
          onChangeContent(e.target.value);
          autoResize();
        }}
        spellCheck={false}
        placeholder={
          message.role === "system"
            ? "System prompt…"
            : message.role === "user"
            ? "User message…"
            : "Assistant response…"
        }
      />
      <button className="msg-delete" onClick={onDelete} title="Delete message">
        ×
      </button>
    </div>
  );
}
