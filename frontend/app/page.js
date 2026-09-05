"use client";

import { useEffect, useRef, useState } from "react";
import {
  Bot,
  CircleAlert,
  FileText,
  Loader2,
  MessageSquarePlus,
  RefreshCw,
  Send,
  Sparkles,
  Square,
  Trash2,
  Upload,
  User,
} from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const starterQuestions = [
  "Which framework is used in background job queueing in CodeStage?",
  "Summarize the uploaded document in five bullets.",
  "What technologies are mentioned in the project?",
];

const welcomeMessage = {
  role: "assistant",
  content: "Ask a question about your uploaded documents. I will answer using the backend RAG pipeline.",
  sources: [],
};

export default function Home() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([welcomeMessage]);
  const [sessionId, setSessionId] = useState("");
  const [health, setHealth] = useState({ status: "checking", label: "Checking backend" });
  const [documents, setDocuments] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [error, setError] = useState("");
  const abortControllerRef = useRef(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const canSend = message.trim().length > 0 && !isSending;

  useEffect(() => {
    const savedSessionId = window.localStorage.getItem("nexusai_session_id");
    if (savedSessionId) {
      setSessionId(savedSessionId);
    }

    checkHealth();
    loadDocuments();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  async function parseJsonResponse(response) {
    const text = await response.text();
    try {
      return text ? JSON.parse(text) : {};
    } catch {
      return { detail: text || response.statusText };
    }
  }

  function friendlyError(detail) {
    const value = String(detail || "Something went wrong.");

    if (value.includes("503") || value.toLowerCase().includes("unavailable")) {
      return "Gemini is busy right now. Please try again in a moment.";
    }

    if (value.toLowerCase().includes("failed to fetch")) {
      return "Cannot reach the backend. Make sure FastAPI is running on port 8000.";
    }

    return value.replace(/^Chat workflow failed:\s*/i, "");
  }

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      if (!response.ok) throw new Error("Backend health check failed");
      const data = await response.json();
      setHealth({
        status: "online",
        label: `${data.project || "Backend"} online`,
      });
    } catch {
      setHealth({
        status: "offline",
        label: "Backend offline",
      });
    }
  }

  async function loadDocuments() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents`);
      const data = await parseJsonResponse(response);

      if (!response.ok) {
        throw new Error(data.detail || "Could not load documents");
      }

      setDocuments(data.documents || []);
    } catch (err) {
      setError(friendlyError(err.message));
    }
  }

  function startNewChat() {
    abortControllerRef.current?.abort();
    window.localStorage.removeItem("nexusai_session_id");
    setSessionId("");
    setError("");
    setIsSending(false);
    setMessages([welcomeMessage]);
  }

  function stopResponse() {
    abortControllerRef.current?.abort();
    setIsSending(false);
    setError("Stopped the current response.");
  }

  async function uploadDocument(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setIsUploading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await parseJsonResponse(response);

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      await loadDocuments();
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `Uploaded ${data.filename} and indexed ${data.chunks_count} chunks.`,
          sources: [],
        },
      ]);
    } catch (err) {
      setError(friendlyError(err.message));
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  async function deleteDocument(documentId) {
    const document = documents.find((item) => item.id === documentId);
    if (!document || deletingId) return;

    const shouldDelete = window.confirm(`Delete ${document.name}?`);
    if (!shouldDelete) return;

    setDeletingId(documentId);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`, {
        method: "DELETE",
      });
      const data = await parseJsonResponse(response);

      if (!response.ok) {
        throw new Error(data.detail || "Delete failed");
      }

      await loadDocuments();
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `Deleted ${document.name}.`,
          sources: [],
        },
      ]);
    } catch (err) {
      setError(friendlyError(err.message));
    } finally {
      setDeletingId(null);
    }
  }

  async function sendMessage(text = message) {
    const question = text.trim();
    if (!question || isSending) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setMessage("");
    setError("");
    setIsSending(true);
    setMessages((current) => [...current, { role: "user", content: question, sources: [] }]);

    try {
      const payload = {
        message: question,
        top_k: 5,
      };

      if (sessionId) {
        payload.session_id = sessionId;
      }

      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const data = await parseJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || "Chat request failed");
      }

      if (data.session_id) {
        setSessionId(data.session_id);
        window.localStorage.setItem("nexusai_session_id", data.session_id);
      }

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer || "No answer returned.",
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      if (err.name === "AbortError") {
        setMessages((current) => [
          ...current,
          {
            role: "assistant",
            content: "Stopped.",
            sources: [],
          },
        ]);
      } else {
        setError(friendlyError(err.message));
      }
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      setIsSending(false);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendMessage();
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={22} />
          </div>
          <div>
            <h1>NexusAI</h1>
            <p>Hybrid RAG assistant</p>
          </div>
        </div>

        <button className="new-chat-button" onClick={startNewChat} type="button">
          <MessageSquarePlus size={18} />
          New chat
        </button>

        <section className="status-panel">
          <div className={`status-dot ${health.status}`} />
          <div>
            <span>Backend</span>
            <strong>{health.label}</strong>
          </div>
        </section>

        <section className="documents-panel">
          <div className="panel-title-row">
            <div>
              <span>Knowledge base</span>
              <strong>{documents.length} documents</strong>
            </div>
            <button onClick={loadDocuments} type="button" aria-label="Refresh documents" title="Refresh documents">
              <RefreshCw size={16} />
            </button>
          </div>

          <input
            ref={fileInputRef}
            className="file-input"
            type="file"
            accept=".pdf,.txt,.md,.csv,.py,.js"
            onChange={(event) => uploadDocument(event.target.files?.[0])}
          />
          <button
            className="upload-button"
            onClick={() => fileInputRef.current?.click()}
            type="button"
            disabled={isUploading}
          >
            {isUploading ? <Loader2 size={17} /> : <Upload size={17} />}
            {isUploading ? "Uploading..." : "Upload document"}
          </button>

          <div className="document-list">
            {documents.length === 0 && <p className="empty-docs">No documents uploaded yet.</p>}
            {documents.map((document) => (
              <article className="document-row" key={document.id}>
                <FileText size={17} />
                <div>
                  <strong title={document.name}>{document.name}</strong>
                  <span>{document.chunk_count} chunks</span>
                </div>
                <button
                  onClick={() => deleteDocument(document.id)}
                  type="button"
                  disabled={deletingId === document.id}
                  aria-label={`Delete ${document.name}`}
                  title={`Delete ${document.name}`}
                >
                  {deletingId === document.id ? <Loader2 size={15} /> : <Trash2 size={15} />}
                </button>
              </article>
            ))}
          </div>
        </section>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <h2>Document Chat</h2>
            <p>{sessionId ? "Current chat" : "New chat"}</p>
          </div>
          <button className="ghost-button" onClick={checkHealth} type="button" aria-label="Refresh backend status" title="Refresh backend status">
            <RefreshCw size={18} />
          </button>
        </header>

        <div className="messages">
          {messages.map((item, index) => (
            <article className={`message ${item.role}`} key={`${item.role}-${index}`}>
              <div className="avatar" aria-hidden="true">
                {item.role === "user" ? <User size={17} /> : <Bot size={17} />}
              </div>
              <div className="bubble">
                <p>{item.content}</p>
                {item.sources?.length > 0 && (
                  <span className="source-pill">{item.sources.length} sources used</span>
                )}
              </div>
            </article>
          ))}

          {isSending && (
            <article className="message assistant">
              <div className="avatar" aria-hidden="true">
                <Bot size={17} />
              </div>
              <div className="bubble loading-bubble">
                <Loader2 size={18} />
                Thinking with your document index...
              </div>
            </article>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="starter-row">
          {starterQuestions.map((question) => (
            <button
              key={question}
              onClick={() => sendMessage(question)}
              type="button"
              disabled={isSending}
            >
              {question}
            </button>
          ))}
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <CircleAlert size={18} />
            <span>{error}</span>
          </div>
        )}

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Ask about your uploaded documents..."
            rows={1}
          />
          {isSending ? (
            <button className="stop-button" onClick={stopResponse} type="button" aria-label="Stop response" title="Stop response">
              <Square size={17} />
            </button>
          ) : (
            <button disabled={!canSend} type="submit" aria-label="Send message" title="Send message">
              <Send size={19} />
            </button>
          )}
        </form>
      </section>
    </main>
  );
}
