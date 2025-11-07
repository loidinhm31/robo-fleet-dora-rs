// Speech recognition and voice command types

export interface SpeechTranscription {
  text: string;
  confidence: number;
  language: string;
  duration_ms: number;
  timestamp: number;
}

export interface SpeechStats {
  total_transcriptions: number;
  avg_confidence: number;
  avg_processing_time_ms: number;
  failed_transcriptions: number;
}
