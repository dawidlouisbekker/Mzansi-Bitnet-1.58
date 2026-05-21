import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

interface TrainingConfig {
  model_path: string;
  dataset_path: string;
  output_path: string;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  lora_rank: number;
  lora_alpha: number;
  grad_accum_steps: number;
  max_seq_len: number;
}

interface ProgressEvent {
  epoch: number;
  step: number;
  total_steps: number;
  loss: number;
}

interface DoneEvent {
  success: boolean;
  error?: string;
}

const DEFAULT_CONFIG: TrainingConfig = {
  model_path: "./models/bitnet-b1.58-2b-4t-bf16",
  dataset_path: "./data/train.jsonl",
  output_path: "./output",
  learning_rate: 0.0002,
  epochs: 3,
  batch_size: 1,
  lora_rank: 16,
  lora_alpha: 32,
  grad_accum_steps: 4,
  max_seq_len: 2048,
};

interface TrainingPanelProps {
  initialDatasetPath?: string;
  onSampleChange?: (idx: number | null) => void;
}

export default function TrainingPanel({ initialDatasetPath, onSampleChange }: TrainingPanelProps) {
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [streamLine, setStreamLine] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<{
    words: string[];
    idx: number;
    header: string;
    timer: ReturnType<typeof setInterval> | null;
  }>({ words: [], idx: 0, header: "", timer: null });

  useEffect(() => {
    if (initialDatasetPath && !running) {
      setConfig((c) => ({ ...c, dataset_path: initialDatasetPath }));
    }
  }, [initialDatasetPath, running]);

  const appendLog = (msg: string) =>
    setLogs((prev) => [...prev.slice(-999), msg]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs, streamLine]);

  function startStream(header: string, text: string) {
    if (streamRef.current.timer) clearInterval(streamRef.current.timer);
    const words = text.split(/\s+/).filter(Boolean);
    streamRef.current = { words, idx: 0, header, timer: null };
    setStreamLine(header);

    streamRef.current.timer = setInterval(() => {
      const s = streamRef.current;
      if (s.idx < s.words.length) {
        const line = s.header + " " + s.words.slice(0, s.idx + 1).join(" ");
        setStreamLine(line);
        s.idx++;
      } else {
        clearInterval(s.timer!);
        s.timer = null;
        setStreamLine((line) => {
          if (line) appendLog(line);
          return "";
        });
      }
    }, 25);
  }

  useEffect(() => {
    const unsubs: UnlistenFn[] = [];

    listen<string>("training://log", (e) => {
      appendLog(e.payload);
    }).then((u) => unsubs.push(u));

    listen<ProgressEvent>("training://progress", (e) => {
      setProgress(e.payload);
    }).then((u) => unsubs.push(u));

    listen<{ epoch: number; sample_idx: number; text: string }>("training://sample", (e) => {
      onSampleChange?.(e.payload.sample_idx);
      startStream(`▶ sample ${e.payload.sample_idx + 1}:`, e.payload.text);
    }).then((u) => unsubs.push(u));

    listen<DoneEvent>("training://done", (e) => {
      setRunning(false);
      onSampleChange?.(null);
      // Flush any in-progress stream
      if (streamRef.current.timer) {
        clearInterval(streamRef.current.timer);
        streamRef.current.timer = null;
      }
      setStreamLine("");
      if (e.payload.success) {
        appendLog("✓ Training complete. Adapter saved.");
      } else {
        appendLog(`✗ Error: ${e.payload.error}`);
        setError(e.payload.error ?? "unknown error");
      }
    }).then((u) => unsubs.push(u));

    return () => unsubs.forEach((u) => u());
  }, []);

  async function startTraining() {
    setError(null);
    setProgress(null);
    setLogs([]);
    setStreamLine("");
    setRunning(true);
    try {
      await invoke("start_training", { config });
    } catch (e: unknown) {
      setError(String(e));
      setRunning(false);
    }
  }

  async function stopTraining() {
    try {
      await invoke("stop_training");
    } catch (e: unknown) {
      setError(String(e));
    }
  }

  function field(
    label: string,
    key: keyof TrainingConfig,
    type: "text" | "number" = "text"
  ) {
    return (
      <label className="field">
        <span>{label}</span>
        <input
          type={type}
          value={String(config[key])}
          disabled={running}
          onChange={(e) =>
            setConfig((c) => ({
              ...c,
              [key]: type === "number" ? Number(e.target.value) : e.target.value,
            }))
          }
        />
      </label>
    );
  }

  const pct = progress
    ? Math.round((progress.step / progress.total_steps) * 100)
    : 0;

  return (
    <div className="panel-right">
      <div className="panel-header">
        <h1>LoRA Training</h1>
      </div>

      <section className="card">
        <h2>Paths</h2>
        {field("Model path", "model_path")}
        {field("Dataset (.jsonl)", "dataset_path")}
        {field("Output dir", "output_path")}
      </section>

      <section className="card">
        <h2>Hyperparameters</h2>
        <div className="grid2">
          {field("Learning rate", "learning_rate", "number")}
          {field("Epochs", "epochs", "number")}
          {field("Batch size", "batch_size", "number")}
          {field("Grad accum steps", "grad_accum_steps", "number")}
          {field("LoRA rank", "lora_rank", "number")}
          {field("LoRA alpha", "lora_alpha", "number")}
          {field("Max seq len", "max_seq_len", "number")}
        </div>
      </section>

      <section className="card controls">
        <button className="btn-start" onClick={startTraining} disabled={running}>
          Start Training
        </button>
        <button className="btn-stop" onClick={stopTraining} disabled={!running}>
          Stop
        </button>
        {error && <p className="error">{error}</p>}
      </section>

      {progress && (
        <section className="card">
          <h2>
            Epoch {progress.epoch} &mdash; Step {progress.step}/
            {progress.total_steps} &mdash; Loss {progress.loss.toFixed(4)}
          </h2>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
        </section>
      )}

      <section className="card log-card">
        <h2>Log</h2>
        <div className="log">
          {logs.map((l, i) => (
            <div key={i} className="log-line">
              {l}
            </div>
          ))}
          {streamLine && (
            <div className="log-line log-streaming">{streamLine}</div>
          )}
          <div ref={logEndRef} />
        </div>
      </section>
    </div>
  );
}
