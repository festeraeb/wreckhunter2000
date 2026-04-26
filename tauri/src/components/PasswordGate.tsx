import { useState, useRef, useEffect } from "react";

const CORRECT = "8106940539";
const SESSION_KEY = "wh2k_unlocked";

interface Props {
  onUnlock: () => void;
}

export function PasswordGate({ onUnlock }: Props) {
  const [value, setValue] = useState("");
  const [error, setError] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    // Restore session
    if (sessionStorage.getItem(SESSION_KEY) === "1") {
      onUnlock();
    }
  }, []);

  const attempt = () => {
    if (value === CORRECT) {
      sessionStorage.setItem(SESSION_KEY, "1");
      onUnlock();
    } else {
      setError(true);
      setValue("");
      setTimeout(() => setError(false), 1200);
    }
  };

  return (
    <div style={{
      position: "fixed", inset: 0,
      background: "rgba(10,14,20,0.97)",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      zIndex: 9999, gap: "1.4rem",
    }}>
      <div style={{ fontSize: "2.4rem" }}>⚓</div>
      <div style={{ color: "#7dd3f0", fontFamily: "monospace", fontSize: "1.1rem", letterSpacing: "0.12em" }}>
        WRECKHUNTER 2000
      </div>
      <div style={{ color: "#888", fontSize: "0.85rem" }}>
        Enter access code for full system access
      </div>
      <input
        ref={inputRef}
        type="password"
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={e => e.key === "Enter" && attempt()}
        placeholder="Access code"
        style={{
          background: "#111",
          border: `1.5px solid ${error ? "#e55" : "#334"}`,
          color: "#eee",
          borderRadius: "6px",
          padding: "0.55rem 1.1rem",
          fontSize: "1rem",
          width: "220px",
          outline: "none",
          textAlign: "center",
          letterSpacing: "0.2em",
          transition: "border-color 0.2s",
        }}
        autoComplete="off"
      />
      {error && (
        <div style={{ color: "#e55", fontSize: "0.82rem" }}>Incorrect code</div>
      )}
      <button
        onClick={attempt}
        style={{
          background: "#1a3a5c",
          color: "#7dd3f0",
          border: "1px solid #2a5a8c",
          borderRadius: "6px",
          padding: "0.5rem 1.6rem",
          cursor: "pointer",
          fontSize: "0.95rem",
          letterSpacing: "0.06em",
        }}
      >
        Unlock
      </button>
    </div>
  );
}
