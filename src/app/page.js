"use client";

import { useChat } from "ai/react";
import { useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const SUGGESTIONS = [
  "What are your technical skills?",
  "Tell me about your work experience",
  "What projects have you worked on?",
  "What is your educational background?",
];

export default function Home() {
  const {
    messages,
    input,
    setInput,
    handleInputChange,
    handleSubmit,
    isLoading,
    error,
  } = useChat({ api: "/api/chat" });

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const sendSuggestion = (suggestion) => {
    if (isLoading) return;
    setInput(suggestion);
    // Submit on next tick after input is set
    setTimeout(() => {
      const form = document.getElementById("chat-form");
      if (form) {
        form.requestSubmit();
      }
    }, 0);
  };

  return (
    <div className="app-container">
      {/* ── Header ──────────────────────────── */}
      <header className="header">
        <h1 className="header-title">Resume Chatbot</h1>
        <p className="header-subtitle">
          ask anything about my professional background
        </p>
      </header>

      {/* ── Chat Area ───────────────────────── */}
      <main className="chat-area" id="chat-area">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="welcome-icon">💼</div>
            <h2 className="welcome-title">Ask Me Anything</h2>
            <p className="welcome-text">
              I&apos;m an AI chatbot trained on my resume. Ask about skills,
              experience, projects, or education, I&apos;ll find the
              answer.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  id={`suggestion-${i}`}
                  className="suggestion-btn"
                  onClick={() => sendSuggestion(s)}
                >
                  → {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`message message--${msg.role}`}
              id={`message-${msg.id}`}
            >
              <span className="message-label">
                {msg.role === "user" ? "You" : "Bot"}
              </span>
              <div className="message-bubble">
                {msg.role === "assistant" ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
            </div>
          ))
        )}

        {/* Typing indicator */}
        {isLoading &&
          (messages.length === 0 ||
            messages[messages.length - 1]?.content === "") && (
            <div className="typing-indicator">
              <div className="typing-dots">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
              <span className="typing-label">Thinking...</span>
            </div>
          )}

        <div ref={chatEndRef} />
      </main>

      {/* ── Error Banner ────────────────────── */}
      {error && (
        <div className="error-banner" id="error-banner">
          {error.message || "Something went wrong. Please try again."}
        </div>
      )}

      {/* ── Input Area ──────────────────────── */}
      <form id="chat-form" className="input-area" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          id="chat-input"
          className="input-field"
          type="text"
          placeholder="Ask about my resume..."
          value={input}
          onChange={handleInputChange}
          disabled={isLoading}
          autoComplete="off"
        />
        <button
          id="send-btn"
          className="send-btn"
          type="submit"
          disabled={!input.trim() || isLoading}
        >
          Send →
        </button>
      </form>

      {/* ── Footer ──────────────────────────── */}
      <footer className="footer">
        Copyright (c) 2026 Yuu. All rights reserved.
      </footer>
    </div>
  );
}
