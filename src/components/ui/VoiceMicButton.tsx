import { useState, useRef } from "react";
import { Mic, MicOff } from "lucide-react";
import { useTranslation } from "react-i18next";

interface VoiceMicButtonProps {
  onTranscript: (text: string) => void;
  className?: string;
}

const LANG_MAP: Record<string, string> = {
  en: "en-IN",
  hi: "hi-IN",
  mr: "mr-IN",
  ta: "ta-IN",
  te: "te-IN",
  bn: "bn-IN",
};

export function VoiceMicButton({ onTranscript, className = "" }: VoiceMicButtonProps) {
  const { i18n } = useTranslation();
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  function toggleListening() {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Voice input is supported in Chrome and Edge browsers.");
      return;
    }

    if (listening && recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = LANG_MAP[i18n.language] || "en-IN";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);
    recognition.onerror = () => setListening(false);

    recognition.onresult = (event: any) => {
      const spokenText = event.results?.[0]?.[0]?.transcript || "";
      if (spokenText.trim()) {
        onTranscript(spokenText.trim());
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  }

  return (
    <button
      type="button"
      onClick={toggleListening}
      title={listening ? "Stop listening" : "Speak using voice"}
      className={`p-2.5 rounded-lg border transition-colors flex items-center justify-center ${
        listening
          ? "bg-red-500 text-white border-red-500 animate-pulse"
          : "bg-white text-deep border-border hover:bg-rose"
      } ${className}`}
    >
      {listening ? <MicOff size={16} /> : <Mic size={16} />}
    </button>
  );
}