import { useState } from "react";
import JsonlEditor from "./JsonL/JsonlEditor";
import TrainingPanel from "./TrainingPanel";
import ChatPage from "./ChatPage";
import "./App.css";

type Tab = "train" | "chat";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("train");
  const [datasetPath, setDatasetPath] = useState<string | undefined>(undefined);
  const [trainingSampleIdx, setTrainingSampleIdx] = useState<number | null>(null);

  return (
    <div className="app-root">
      <div className="tab-bar">
        <span className="tab-bar-brand">MzansiLM</span>
        <button
          className={`tab-btn${activeTab === "train" ? " tab-btn--active" : ""}`}
          onClick={() => setActiveTab("train")}
        >
          Train
        </button>
        <button
          className={`tab-btn${activeTab === "chat" ? " tab-btn--active" : ""}`}
          onClick={() => setActiveTab("chat")}
        >
          Chat
        </button>
      </div>

      {activeTab === "train" ? (
        <div className="app-shell">
          <JsonlEditor
            onPathChange={setDatasetPath}
            trainingSampleIdx={trainingSampleIdx}
          />
          <TrainingPanel
            initialDatasetPath={datasetPath}
            onSampleChange={setTrainingSampleIdx}
          />
        </div>
      ) : (
        <ChatPage />
      )}
    </div>
  );
}
