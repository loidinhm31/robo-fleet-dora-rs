import React, { useEffect, useState } from "react";
import { SpeechTranscription } from "../types";
import { ChevronDown, Mic, MicOff, Volume2 } from "lucide-react";
import { DraggablePanel } from "./organisms";
import { StatusBadge } from "./atoms";

interface TranscriptionDisplayProps {
  transcription: SpeechTranscription | null;
  isAudioActive: boolean;
  maxHistory?: number;
  onStartAudio?: () => void;
  onStopAudio?: () => void;
}

interface TranscriptionHistoryItem extends SpeechTranscription {
  id: string;
}

export const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({
  transcription,
  isAudioActive,
  maxHistory = 5,
  onStartAudio,
  onStopAudio,
}) => {
  const [history, setHistory] = useState<TranscriptionHistoryItem[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (transcription) {
      const newItem: TranscriptionHistoryItem = {
        ...transcription,
        id: `${transcription.timestamp}-${Math.random()}`,
      };

      setHistory((prev) => {
        const updated = [newItem, ...prev];
        return updated.slice(0, maxHistory);
      });

      setIsAnimating(true);
      const timer = setTimeout(() => setIsAnimating(false), 300);
      return () => clearTimeout(timer);
    }
  }, [transcription, maxHistory]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-400";
    if (confidence >= 0.6) return "text-yellow-400";
    return "text-orange-400";
  };

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) return "bg-green-500/20 text-green-400";
    if (confidence >= 0.6) return "bg-yellow-500/20 text-yellow-400";
    return "bg-orange-500/20 text-orange-400";
  };

  // Collapsed mini indicator
  const collapsedContent = (
    <button className="group flex items-center gap-2 px-3 py-1.5 bg-slate-900/95 backdrop-blur-md border border-slate-700/50 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105 drag-handle cursor-move">
      {isAudioActive ? (
        <>
          <Mic className="w-3.5 h-3.5 text-green-400 animate-pulse" />
          <span className="text-[10px] font-bold text-green-400 uppercase tracking-wide">
            Live
          </span>
        </>
      ) : (
        <>
          <MicOff className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">
            Off
          </span>
        </>
      )}
      <ChevronDown className="w-3 h-3 text-slate-400 group-hover:text-slate-300" />
    </button>
  );

  return (
    <DraggablePanel
      title="Speech Transcription"
      isVisible={isVisible}
      onToggleVisible={() => setIsVisible(!isVisible)}
      collapsedContent={collapsedContent}
      initialPosition={{ x: 0, y: 70 }}
      className="w-96 max-h-[40vh]"
      contentClassName="flex-1 overflow-y-auto custom-scrollbar p-0"
      showControls={true}
    >
      {/* Custom Header */}
      <div className="flex items-center justify-between px-2.5 py-1.5 border-b border-slate-700/50 bg-gradient-to-r from-slate-800/90 to-slate-900/90 -mt-4">
        <div className="flex items-center gap-1.5">
          <StatusBadge
            variant={isAudioActive ? "online" : "offline"}
            label={isAudioActive ? "Live" : "Off"}
            animated={isAudioActive}
          />
        </div>

        {/* Audio control button */}
        {onStartAudio && onStopAudio && (
          <button
            onClick={isAudioActive ? onStopAudio : onStartAudio}
            className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wide transition-all ${
              isAudioActive
                ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
                : "bg-green-500/20 text-green-400 hover:bg-green-500/30"
            }`}
          >
            {isAudioActive ? "Stop" : "Start"}
          </button>
        )}
      </div>

      {/* Current Transcription */}
      {transcription && (
        <div
          className={`px-2.5 py-2 border-b border-slate-700/30 transition-all duration-200 ${
            isAnimating ? "bg-blue-500/10" : "bg-slate-800/30"
          }`}
        >
          <p className="text-xs font-medium text-slate-100 leading-tight mb-1">
            {transcription.text}
          </p>
          <div className="flex items-center gap-1.5 text-[10px]">
            <span
              className={`px-1 py-0.5 rounded text-[10px] font-bold ${getConfidenceBadge(transcription.confidence)}`}
            >
              {(transcription.confidence * 100).toFixed(0)}%
            </span>
            <span className="text-slate-500">
              {new Date(transcription.timestamp).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
            {transcription.duration_ms && (
              <span className="text-slate-500 flex items-center gap-0.5">
                <Volume2 className="w-2.5 h-2.5" />
                {(transcription.duration_ms / 1000).toFixed(1)}s
              </span>
            )}
          </div>
        </div>
      )}

      {/* Transcription History */}
      {history.length > 1 && (
        <div className="divide-y divide-slate-700/20">
          {history.slice(1, 4).map((item, index) => (
            <div
              key={item.id}
              className="px-2.5 py-1.5 hover:bg-slate-800/20 transition-colors"
              style={{ opacity: Math.max(0.5, 1 - index * 0.15) }}
            >
              <p className="text-[11px] text-slate-300 leading-tight mb-0.5 line-clamp-2">
                {item.text}
              </p>
              <div className="flex items-center gap-1.5">
                <span
                  className={`text-[10px] font-bold ${getConfidenceColor(item.confidence)}`}
                >
                  {(item.confidence * 100).toFixed(0)}%
                </span>
                <span className="text-[10px] text-slate-600">
                  {new Date(item.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!transcription && history.length === 0 && (
        <div className="px-2.5 py-6 text-center">
          <Mic className="w-8 h-8 text-slate-600 mx-auto mb-1.5 opacity-50" />
          <p className="text-[10px] text-slate-500 font-medium">
            {isAudioActive ? "Listening..." : "Start to transcribe"}
          </p>
        </div>
      )}
    </DraggablePanel>
  );
};

export default TranscriptionDisplay;
