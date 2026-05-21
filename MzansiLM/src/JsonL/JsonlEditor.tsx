import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import JsonlEntryCard from "./JsonlEntry";
import type { JsonlEntry, Message } from "../types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function parseJsonl(text: string): JsonlEntry[] {
  return text
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => {
      try {
        const obj = JSON.parse(line);
        return { id: uid(), messages: (obj.messages ?? []) as Message[], raw: line };
      } catch (e) {
        return { id: uid(), messages: [], raw: line, parseError: String(e) };
      }
    });
}

function entriesToJsonl(entries: JsonlEntry[]): string {
  return (
    entries
      .map((e) =>
        e.parseError ? e.raw : JSON.stringify({ messages: e.messages })
      )
      .join("\n") + "\n"
  );
}

const BLANK_ENTRY: () => JsonlEntry = () => ({
  id: uid(),
  messages: [
    { role: "user", content: "" },
    { role: "assistant", content: "" },
  ],
  raw: "",
});

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface JsonlEditorProps {
  onPathChange?: (path: string) => void;
  trainingSampleIdx?: number | null;
}

export default function JsonlEditor({ onPathChange, trainingSampleIdx }: JsonlEditorProps) {
  const [filePath, setFilePath] = useState("./data/train.jsonl");
  const [entries, setEntries] = useState<JsonlEntry[]>([]);
  const [savedJson, setSavedJson] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const currentJson = entriesToJsonl(entries);
  const dirty = loaded && currentJson !== savedJson;

  const byteSize = new TextEncoder().encode(currentJson).length;
  const sizeLabel =
    byteSize < 1024
      ? `${byteSize} B`
      : byteSize < 1024 * 1024
      ? `${(byteSize / 1024).toFixed(1)} KB`
      : `${(byteSize / (1024 * 1024)).toFixed(2)} MB`;

  // Auto-load on mount
  useEffect(() => {
    if (filePath) loadFile(filePath);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Clear save status after 2s
  useEffect(() => {
    if (!saveStatus) return;
    const t = setTimeout(() => setSaveStatus(null), 2000);
    return () => clearTimeout(t);
  }, [saveStatus]);

  async function loadFile(path: string) {
    setLoadError(null);
    try {
      const text = await invoke<string>("read_file", { path });
      const parsed = parseJsonl(text);
      setEntries(parsed);
      setSavedJson(entriesToJsonl(parsed));
      setLoaded(true);
      onPathChange?.(path);
    } catch (e) {
      setEntries([]);
      setSavedJson("");
      setLoaded(true);
      setLoadError(`${String(e)} — starting with empty file`);
      onPathChange?.(path);
    }
  }

  async function saveFile() {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    const jsonl = entriesToJsonl(entries);
    try {
      await invoke("write_file", { path: filePath, content: jsonl });
      setSavedJson(jsonl);
      setSaveStatus("Saved");
    } catch (e) {
      setSaveStatus(`Error: ${String(e)}`);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      saveFile();
    }
  }

  function updateEntry(updated: JsonlEntry) {
    setEntries((prev) => prev.map((en) => (en.id === updated.id ? updated : en)));
  }

  function deleteEntry(id: string) {
    setEntries((prev) => prev.filter((en) => en.id !== id));
  }

  function addEntry() {
    setEntries((prev) => [...prev, BLANK_ENTRY()]);
  }

  function handlePathKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") loadFile(filePath);
  }

  return (
    <div className="panel-left" onKeyDown={handleKeyDown} tabIndex={-1}>
      {/* Header */}
      <div className="editor-header">
        <span className="editor-title">JSONL Editor</span>
        {dirty && <span className="dirty-dot" title="Unsaved changes">●</span>}
        <div style={{ display: "flex", gap: "0.4rem", marginLeft: "auto" }}>
          <button className="btn-sm btn-ghost" onClick={addEntry}>
            + Example
          </button>
          <button
            className="btn-sm btn-start"
            onClick={saveFile}
            disabled={!dirty}
            title="Save (Ctrl+S)"
          >
            {saveStatus === "Saved" ? "Saved ✓" : "Save"}
          </button>
        </div>
      </div>

      {/* File bar */}
      <div className="editor-file-bar">
        <span className="editor-file-label">Path</span>
        <input
          className="editor-path-input"
          type="text"
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          onKeyDown={handlePathKeyDown}
          placeholder="./data/train.jsonl"
          spellCheck={false}
        />
        <button className="btn-sm btn-ghost" onClick={() => loadFile(filePath)}>
          Load
        </button>
      </div>

      {/* Error banner */}
      {loadError && (
        <div className="editor-banner editor-banner-warn">{loadError}</div>
      )}

      {/* Entry list */}
      <div className="entry-list">
        {entries.length === 0 && loaded && (
          <p style={{ color: "#444", fontSize: "0.8rem", textAlign: "center", marginTop: "2rem" }}>
            No entries — click "+ Example" to add one.
          </p>
        )}
        {entries.map((entry, i) => (
          <JsonlEntryCard
            key={entry.id}
            entry={entry}
            index={i}
            isTraining={trainingSampleIdx === i}
            onChange={updateEntry}
            onDelete={deleteEntry}
          />
        ))}
      </div>

      {/* Status bar */}
      <div className="editor-status-bar">
        <span>{entries.length} example{entries.length !== 1 ? "s" : ""}</span>
        <span>{sizeLabel}</span>
        {saveStatus && (
          <span className={saveStatus.startsWith("Error") ? "editor-error" : "editor-success"}>
            {saveStatus}
          </span>
        )}
        <span style={{ marginLeft: "auto", color: "#3a3a3a" }}>Ctrl+S to save</span>
      </div>
    </div>
  );
}
