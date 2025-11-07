// UI-specific types

export interface LogEntry {
  timestamp: string;
  message: string;
  type: "info" | "success" | "error" | "warning";
}

export interface ConnectionState {
  isConnected: boolean;
  clientId: string | null;
  commandsSent: number;
  commandsReceived: number;
}
