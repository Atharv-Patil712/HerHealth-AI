import { useEffect, useRef, useState } from "react";
import { Send } from "lucide-react";
import { api } from "../lib/api";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
}

interface ChatResponse {
  content: string;
}

const TOPICS = [
  { label: "PCOS", prompt: "Can you explain PCOS/PCOD — what it is, common symptoms, and when I should see a doctor?" },
  { label: "Anemia", prompt: "What causes anemia in women, what are the symptoms, and how is it usually managed?" },
  { label: "Periods & Cycles", prompt: "What counts as a normal menstrual cycle, and what changes are worth getting checked?" },
  { label: "Fertility", prompt: "How does ovulation work, and what are the basics of understanding fertility?" },
  { label: "Thyroid", prompt: "How do thyroid disorders affect women's health and periods?" },
  { label: "Endometriosis", prompt: "What is endometriosis, what are its symptoms, and how is it diagnosed?" },
  { label: "Menopause", prompt: "What happens during perimenopause and menopause, and what symptoms are common?" },
  { label: "UTIs & Infections", prompt: "What are common signs of UTIs and vaginal infections, and when should I see a doctor?" },
];

function CycleRing({ size = 28, spinning = false }: { size?: number; spinning?: boolean }) {
  const dots = Array.from({ length: 8 }, (_, i) => i);
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 4;

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className={spinning ? "ws-ring ws-ring-spin" : "ws-ring"}
      aria-hidden="true"
    >
      {dots.map((i) => {
        const angle = (i / dots.length) * 2 * Math.PI - Math.PI / 2;
        const isMain = i % 2 === 0;
        return (
          <circle
            key={i}
            cx={cx + r * Math.cos(angle)}
            cy={cy + r * Math.sin(angle)}
            r={isMain ? 2.4 : 1.5}
            fill={isMain ? "var(--ws-berry)" : "var(--ws-gold)"}
            opacity={isMain ? 1 : 0.7}
          />
        );
      })}
      <circle cx={cx} cy={cy} r={1.8} fill="var(--ws-sage)" />
    </svg>
  );
}

export default function Chatbot() {
  const idRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 0,
      role: "assistant",
      content:
        "Hi, I'm Wellspring. I'm here to talk through periods, PCOS, anemia, fertility, and other reproductive and women's health topics.\n\nAsk me anything, or tap a topic below to start.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading]);

  const nextId = () => {
    idRef.current += 1;
    return idRef.current;
  };

  const send = async (raw?: string) => {
    const text = (raw ?? input).trim();
    if (!text || isLoading) return;

    const userMessage: Message = { id: nextId(), role: "user", content: text };
    const history = [...messages, userMessage];
    setMessages(history);
    setInput("");
    setIsLoading(true);

    try {
      const response = await api.post<ChatResponse>("/api/chat", {
        messages: history.map(({ role, content }) => ({ role, content })),
      });
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "assistant", content: response.content },
      ]);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "I couldn't connect just now.";
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "assistant",
          content: `I couldn't connect just now. ${message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="ws-app">
      <style>{`
        .ws-app, .ws-app *, .ws-app *::before, .ws-app *::after { box-sizing: border-box; }
        .ws-app {
          --ws-bg-page: #ede1de; --ws-bg: #fbf6f3; --ws-panel: #f3e6e1;
          --ws-ink: #2e1a2b; --ws-ink-soft: #6b5560; --ws-berry: #9c3b54;
          --ws-berry-dark: #7c2e42; --ws-sage: #6e8266; --ws-gold: #b98a2e;
          --ws-line: #e6d8d2;
          min-height: calc(100vh - 64px); width: 100%; display: flex;
          justify-content: center; align-items: flex-start; background: var(--ws-bg-page);
          padding: 28px 12px; font-family: "Nunito Sans", sans-serif;
        }
        .ws-card {
          width: 100%; max-width: 480px; height: min(720px, calc(100vh - 120px));
          min-height: 520px; background: var(--ws-bg); border: 1px solid var(--ws-line);
          border-radius: 24px; box-shadow: 0 20px 50px -20px rgba(46,26,43,.35);
          display: flex; flex-direction: column; overflow: hidden;
        }
        .ws-header { padding: 18px 20px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid var(--ws-line); }
        .ws-header-text { display: flex; flex-direction: column; gap: 2px; }
        .ws-title { font-family: Georgia, serif; font-size: 21px; font-weight: 600; color: var(--ws-ink); }
        .ws-subtitle { font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--ws-ink-soft); }
        .ws-topics { display: flex; gap: 8px; padding: 12px 16px; overflow-x: auto; background: var(--ws-panel); border-bottom: 1px solid var(--ws-line); }
        .ws-chip { flex: none; padding: 7px 14px; border-radius: 999px; border: 1px solid var(--ws-line); background: var(--ws-bg); color: var(--ws-ink); font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap; }
        .ws-chip:not(:disabled):hover { background: var(--ws-berry); color: #fff; border-color: var(--ws-berry); }
        .ws-chip:disabled { opacity: .5; cursor: not-allowed; }
        .ws-chat { flex: 1 1 auto; overflow-y: auto; padding: 18px 16px; display: flex; flex-direction: column; gap: 14px; }
        .ws-row { display: flex; gap: 8px; align-items: flex-end; }
        .ws-row.user { justify-content: flex-end; }
        .ws-row.bot { justify-content: flex-start; }
        .ws-avatar { flex: none; width: 26px; height: 26px; border-radius: 50%; background: var(--ws-bg); border: 1px solid var(--ws-line); display: flex; align-items: center; justify-content: center; }
        .ws-bubble { max-width: 78%; padding: 11px 14px; font-size: 14.5px; line-height: 1.55; white-space: pre-wrap; }
        .ws-bubble.bot { background: #fff; color: var(--ws-ink); border: 1px solid var(--ws-line); border-radius: 16px 16px 16px 4px; }
        .ws-bubble.user { background: var(--ws-berry); color: #fff; border-radius: 16px 16px 4px 16px; }
        .ws-thinking-text { font-size: 11px; color: var(--ws-ink-soft); letter-spacing: .04em; }
        .ws-ring-spin { animation: ws-spin 2.2s linear infinite; transform-origin: center; }
        @keyframes ws-spin { to { transform: rotate(360deg); } }
        .ws-inputbar { display: flex; align-items: center; gap: 8px; padding: 12px 14px; border-top: 1px solid var(--ws-line); }
        .ws-input { flex: 1 1 auto; padding: 10px 14px; border-radius: 999px; border: 1px solid var(--ws-line); font-size: 14px; color: var(--ws-ink); background: #fff; outline: none; }
        .ws-input:focus-visible { border-color: var(--ws-berry); box-shadow: 0 0 0 3px rgba(156,59,84,.15); }
        .ws-send { flex: none; width: 38px; height: 38px; border-radius: 50%; border: none; background: var(--ws-berry); color: #fff; display: flex; align-items: center; justify-content: center; cursor: pointer; }
        .ws-send:disabled { opacity: .45; cursor: not-allowed; }
        .ws-disclaimer { padding: 8px 16px 14px; font-size: 11px; color: var(--ws-ink-soft); text-align: center; line-height: 1.4; margin: 0; }
        @media (prefers-reduced-motion: reduce) { .ws-ring-spin { animation: none; } }
      `}</style>

      <div className="ws-card">
        <header className="ws-header">
          <CycleRing size={30} />
          <div className="ws-header-text">
            <span className="ws-title">Wellspring</span>
            <span className="ws-subtitle">Women's Health Companion</span>
          </div>
        </header>

        <div className="ws-topics" aria-label="Quick topics">
          {TOPICS.map((topic) => (
            <button
              key={topic.label}
              type="button"
              className="ws-chip"
              onClick={() => send(topic.prompt)}
              disabled={isLoading}
            >
              {topic.label}
            </button>
          ))}
        </div>

        <div className="ws-chat" aria-live="polite">
          {messages.map((message) => (
            <div key={message.id} className={`ws-row ${message.role === "user" ? "user" : "bot"}`}>
              {message.role === "assistant" && (
                <div className="ws-avatar"><CycleRing size={16} /></div>
              )}
              <div className={`ws-bubble ${message.role === "user" ? "user" : "bot"}`}>
                {message.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="ws-row bot">
              <div className="ws-avatar"><CycleRing size={16} spinning /></div>
              <div className="ws-bubble bot"><span className="ws-thinking-text">Wellspring is thinking…</span></div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="ws-inputbar">
          <input
            className="ws-input"
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                send();
              }
            }}
            placeholder="Ask about symptoms, cycles, conditions…"
            aria-label="Type your question"
            disabled={isLoading}
          />
          <button
            type="button"
            className="ws-send"
            onClick={() => send()}
            disabled={isLoading || !input.trim()}
            aria-label="Send message"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="ws-disclaimer">
          General health information, not a diagnosis. If something feels urgent, contact a doctor or local emergency services.
        </p>
      </div>
    </div>
  );
}
