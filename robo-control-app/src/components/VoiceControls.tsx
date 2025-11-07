import React, { useCallback, useEffect, useRef, useState } from "react";
import { Socket } from "socket.io-client";
import { Mic, Volume2, Send, Radio, Headphones, AlertCircle, Shield, ChevronDown } from "lucide-react";
import { DraggablePanel } from "./organisms";
import { InputWithAction } from "./molecules";
import { IconBadge, StatusBadge } from "./atoms";

interface VoiceControlsProps {
  socket: Socket | null;
  isConnected: boolean;
  onLog?: (message: string, type?: "info" | "success" | "error" | "warning") => void;
}

type VoiceMode = "idle" | "voice_commands" | "walkie_talkie";

export const VoiceControls: React.FC<VoiceControlsProps> = ({
  socket,
  isConnected,
  onLog,
}) => {
  // TTS state
  const [ttsText, setTtsText] = useState("");
  const [isSendingTTS, setIsSendingTTS] = useState(false);

  // Voice mode state
  const [voiceMode, setVoiceMode] = useState<VoiceMode>("idle");
  const [audioLevel, setAudioLevel] = useState(0);
  const [permissionError, setPermissionError] = useState<string>("");

  // Visibility state
  const [isVisible, setIsVisible] = useState(false);

  // Audio refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Send TTS command
  const sendTTS = useCallback((text: string) => {
    if (!isConnected || !socket || !text.trim()) {
      onLog?.("Cannot send TTS - not connected or empty text", "error");
      return;
    }

    setIsSendingTTS(true);
    socket.emit("tts_command", { text: text.trim() });
    onLog?.(`TTS: "${text.trim()}"`, "success");

    setTimeout(() => {
      setIsSendingTTS(false);
    }, 300);
  }, [isConnected, socket, onLog]);

  // Visualize audio level
  const visualizeAudioLevel = useCallback(() => {
    if (!analyserRef.current) return;

    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);

    const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
    setAudioLevel(average / 255);

    animationFrameRef.current = requestAnimationFrame(visualizeAudioLevel);
  }, []);

  // Request microphone permission
  const requestMicrophonePermission = useCallback(async (): Promise<MediaStream | null> => {
    setPermissionError("");

    try {
      if (!window.isSecureContext && window.location.hostname !== "localhost") {
        setPermissionError("Microphone requires HTTPS or localhost");
        onLog?.("Microphone requires HTTPS or localhost", "error");
        return null;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1,
        },
      });

      return stream;
    } catch (error) {
      console.error("Microphone permission error:", error);

      if (error instanceof Error) {
        if (error.name === "NotAllowedError") {
          setPermissionError(
            "Microphone permission denied. Please allow microphone access in your browser settings."
          );
        } else if (error.name === "NotFoundError") {
          setPermissionError("No microphone found. Please connect a microphone.");
          onLog?.("No microphone device found", "error");
        } else if (error.name === "NotReadableError") {
          setPermissionError("Microphone is already in use by another application.");
          onLog?.("Microphone already in use", "error");
        } else {
          setPermissionError(`Microphone error: ${error.message}`);
          onLog?.(`Microphone error: ${error.message}`, "error");
        }
      }

      return null;
    }
  }, [onLog]);

  // Create AudioWorklet processor
  const createAudioWorkletProcessor = useCallback((bufferSize: number, processorName: string) => {
    const processorCode = `
      class ${processorName} extends AudioWorkletProcessor {
        constructor() {
          super();
          this.bufferSize = ${bufferSize};
          this.buffer = new Float32Array(this.bufferSize);
          this.bufferIndex = 0;
        }

        process(inputs, outputs, parameters) {
          const input = inputs[0];
          if (input.length > 0) {
            const channelData = input[0];

            for (let i = 0; i < channelData.length; i++) {
              this.buffer[this.bufferIndex] = channelData[i];
              this.bufferIndex++;

              if (this.bufferIndex >= this.bufferSize) {
                this.port.postMessage({
                  type: 'audio-data',
                  audioData: new Float32Array(this.buffer)
                });

                this.bufferIndex = 0;
                this.buffer.fill(0);
              }
            }
          }

          return true;
        }
      }

      registerProcessor('${processorName}', ${processorName});
    `;

    const blob = new Blob([processorCode], { type: "application/javascript" });
    return URL.createObjectURL(blob);
  }, []);

  // Start voice command mode
  const startVoiceCommands = useCallback(async () => {
    if (!isConnected || !socket) {
      onLog?.("Cannot start voice commands - not connected", "error");
      return;
    }

    const stream = await requestMicrophonePermission();
    if (!stream) return;

    try {
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const processorUrl = createAudioWorkletProcessor(4096, "VoiceCommandProcessor");
      await audioContext.audioWorklet.addModule(processorUrl);

      const workletNode = new AudioWorkletNode(audioContext, "VoiceCommandProcessor");
      source.connect(workletNode);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      mediaStreamRef.current = stream;
      setVoiceMode("voice_commands");
      visualizeAudioLevel();

      onLog?.("Voice commands started", "success");
    } catch (error) {
      console.error("Voice command error:", error);
      onLog?.("Failed to start voice commands", "error");
      stream.getTracks().forEach((track) => track.stop());
    }
  }, [isConnected, socket, onLog, requestMicrophonePermission, createAudioWorkletProcessor, visualizeAudioLevel]);

  // Start walkie-talkie mode
  const startWalkieTalkie = useCallback(async () => {
    if (!isConnected || !socket) {
      onLog?.("Cannot start walkie-talkie - not connected", "error");
      return;
    }

    const stream = await requestMicrophonePermission();
    if (!stream) return;

    try {
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const processorUrl = createAudioWorkletProcessor(800, "WalkieTalkieProcessor");
      await audioContext.audioWorklet.addModule(processorUrl);

      const workletNode = new AudioWorkletNode(audioContext, "WalkieTalkieProcessor");
      workletNode.port.onmessage = (event) => {
        if (event.data.type === "audio-data") {
          socket.emit("audio_stream", { audio_data: Array.from(event.data.audioData) });
        }
      };

      source.connect(workletNode);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      mediaStreamRef.current = stream;
      setVoiceMode("walkie_talkie");
      visualizeAudioLevel();

      onLog?.("Walkie-talkie started", "success");
    } catch (error) {
      console.error("Walkie-talkie error:", error);
      onLog?.("Failed to start walkie-talkie", "error");
      stream.getTracks().forEach((track) => track.stop());
    }
  }, [isConnected, socket, onLog, requestMicrophonePermission, createAudioWorkletProcessor, visualizeAudioLevel]);

  // Stop current voice mode
  const stopVoiceMode = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    analyserRef.current = null;
    processorRef.current = null;
    setVoiceMode("idle");
    setAudioLevel(0);
    onLog?.("Voice mode stopped", "info");
  }, [onLog]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopVoiceMode();
    };
  }, [stopVoiceMode]);

  // Collapsed content
  const collapsedContent = (
    <button className="group flex items-center gap-2 px-3 py-1.5 bg-slate-900/95 backdrop-blur-md border border-slate-700/50 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105 drag-handle cursor-move">
      <Volume2 className="w-3.5 h-3.5 text-orange-400" />
      <span className="text-[10px] font-bold text-white uppercase tracking-wide">Voice</span>
      <ChevronDown className="w-3 h-3 text-slate-400 group-hover:text-slate-300" />
    </button>
  );

  return (
    <DraggablePanel
      title="VOICE COMMUNICATION"
      isVisible={isVisible}
      onToggleVisible={() => setIsVisible(!isVisible)}
      initialPosition={{ x: 15, y: 55 }}
      collapsedContent={collapsedContent}
      className="max-w-md"
      contentClassName="flex-1 overflow-y-auto custom-scrollbar p-0"
      showControls={true}
    >
      <div className="space-y-3">
        {/* Connection Warning */}
        {!isConnected && (
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-2 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-yellow-400" />
            <span className="text-xs text-yellow-400">Not connected to server</span>
          </div>
        )}

        {/* Permission Error */}
        {permissionError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-2 flex items-center gap-2">
            <Shield className="w-4 h-4 text-red-400" />
            <span className="text-xs text-red-400">{permissionError}</span>
          </div>
        )}

        {/* TTS Section */}
        <div className="glass-card-light rounded-xl p-3 space-y-2">
          <div className="flex items-center gap-2">
            <Volume2 className="w-4 h-4 text-orange-400" />
            <h3 className="text-sm font-semibold text-white">Text-to-Speech</h3>
          </div>
          <InputWithAction
            value={ttsText}
            onChange={setTtsText}
            onSubmit={sendTTS}
            placeholder="Type message to speak..."
            icon={Send}
            disabled={!isConnected || isSendingTTS}
          />
        </div>

        {/* Voice Modes Section */}
        <div className="glass-card-light rounded-xl p-3 space-y-3">
          <div className="flex items-center gap-2">
            <Radio className="w-4 h-4 text-green-400" />
            <h3 className="text-sm font-semibold text-white">Voice Modes</h3>
            {voiceMode !== "idle" && (
              <IconBadge
                icon={voiceMode === "voice_commands" ? Mic : Headphones}
                label={voiceMode === "voice_commands" ? "Commands" : "Walkie"}
                color="text-green-400"
                size="sm"
                animated
              />
            )}
          </div>

          {/* Voice Commands */}
          <div className="glass-card-light rounded-xl p-3 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Mic className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-semibold text-white">Voice Commands</span>
              </div>
              <StatusBadge
                variant={voiceMode === "voice_commands" ? "online" : "offline"}
                animated={voiceMode === "voice_commands"}
              />
            </div>
            <p className="text-xs text-white/60">Use voice to control the rover</p>
            <button
              onClick={() => {
                if (voiceMode === "voice_commands") {
                  stopVoiceMode();
                } else {
                  if (voiceMode !== "idle") stopVoiceMode();
                  startVoiceCommands();
                }
              }}
              disabled={!isConnected}
              className={`w-full py-2 px-4 rounded-lg font-semibold transition-all duration-300 ${
                voiceMode === "voice_commands"
                  ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
                  : "bg-green-500/20 text-green-400 hover:bg-green-500/30"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {voiceMode === "voice_commands" ? "Stop" : "Start"}
            </button>
          </div>

          {/* Walkie-Talkie */}
          <div className="glass-card-light rounded-xl p-3 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Headphones className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-semibold text-white">Walkie-Talkie</span>
              </div>
              <StatusBadge
                variant={voiceMode === "walkie_talkie" ? "online" : "offline"}
                animated={voiceMode === "walkie_talkie"}
              />
            </div>
            <p className="text-xs text-white/60">Stream audio directly to rover</p>
            <button
              onClick={() => {
                if (voiceMode === "walkie_talkie") {
                  stopVoiceMode();
                } else {
                  if (voiceMode !== "idle") stopVoiceMode();
                  startWalkieTalkie();
                }
              }}
              disabled={!isConnected}
              className={`w-full py-2 px-4 rounded-lg font-semibold transition-all duration-300 ${
                voiceMode === "walkie_talkie"
                  ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
                  : "bg-green-500/20 text-green-400 hover:bg-green-500/30"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {voiceMode === "walkie_talkie" ? "Stop" : "Start"}
            </button>
          </div>

          {/* Audio Level Visualization */}
          {voiceMode !== "idle" && (
            <div className="mt-2">
              <div className="flex justify-between text-xs text-white/60 mb-1">
                <span>Audio Level</span>
                <span>{(audioLevel * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full h-2 bg-slate-700/50 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-green-400 to-emerald-500 transition-all duration-100"
                  style={{ width: `${audioLevel * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Help Text */}
        <div className="text-xs text-white/40 space-y-1">
          <p>• <strong>Voice Commands:</strong> Say "move forward", "turn left", "track person", etc.</p>
          <p>• <strong>Walkie-Talkie:</strong> Direct audio streaming for communication</p>
          <p>• <strong>TTS:</strong> Type text for rover to speak</p>
        </div>
      </div>
    </DraggablePanel>
  );
};

export default VoiceControls;
