export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface JsonlEntry {
  id: string;
  messages: Message[];
  raw: string;
  parseError?: string;
}
