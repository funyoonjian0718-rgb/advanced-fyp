import { useEffect, useState, type FormEvent } from "react";
import { isAxiosError } from "axios";
import { useSearchParams } from "react-router-dom";

import {
    getChatSession,
    sendMessage,
    type ChatCitation,
    type ChatMessage,
} from "../services/chat";

function apiError(error: unknown): string {
    if (isAxiosError<{ detail?: unknown }>(error)) {
        const detail = error.response?.data?.detail;
        if (typeof detail === "string") {
            return detail;
        }
    }
    return "The request failed. Please try again.";
}

function Citations({ items }: { items: ChatCitation[] }) {
    if (!items.length) {
        return null;
    }

    return (
        <details className="mt-3 text-sm">
            <summary className="cursor-pointer font-medium text-blue-700">
                Sources ({items.length})
            </summary>
            <ul className="mt-2 space-y-2">
                {items.map((citation) => (
                    <li
                        key={citation.evidence_id}
                        className="rounded border border-gray-200 bg-white p-3"
                    >
                        <p className="font-medium">{citation.source_title}</p>
                        <p className="text-gray-600">
                            {citation.source_type} · source authority{" "}
                            {Math.round(citation.authority_score * 100)}%
                        </p>
                        <p className="mt-1 text-gray-700">{citation.excerpt}</p>
                        {citation.source_url && (
                            <a
                                href={citation.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="mt-1 inline-block text-blue-700 underline"
                            >
                                View source
                            </a>
                        )}
                    </li>
                ))}
            </ul>
        </details>
    );
}

function Chat() {
    const [searchParams, setSearchParams] = useSearchParams();
    const rawSessionId = searchParams.get("session_id");
    const sessionId = rawSessionId ? Number(rawSessionId) : undefined;
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [draft, setDraft] = useState("");
    const [error, setError] = useState("");
    const [loadingHistory, setLoadingHistory] = useState(false);
    const [sending, setSending] = useState(false);

    useEffect(() => {
        let active = true;

        async function loadSession() {
            if (!sessionId || !Number.isSafeInteger(sessionId)) {
                setMessages([]);
                setError("");
                return;
            }

            setLoadingHistory(true);
            setError("");
            try {
                const session = await getChatSession(sessionId);
                if (active) {
                    setMessages(session.messages);
                }
            } catch (loadError: unknown) {
                if (active) {
                    setError(apiError(loadError));
                }
            } finally {
                if (active) {
                    setLoadingHistory(false);
                }
            }
        }

        void loadSession();
        return () => {
            active = false;
        };
    }, [sessionId]);

    async function handleSubmit(event: FormEvent<HTMLFormElement>) {
        event.preventDefault();
        const message = draft.trim();
        if (!message || sending) {
            return;
        }

        setError("");
        setSending(true);
        const temporaryMessage: ChatMessage = {
            id: Date.now(),
            role: "user",
            content: message,
            citations: [],
            warnings: [],
            created_at: new Date().toISOString(),
        };
        setMessages((current) => [...current, temporaryMessage]);

        try {
            const result = await sendMessage(message, sessionId);
            const assistantMessage: ChatMessage = {
                id: Date.now() + 1,
                role: "assistant",
                content: result.answer,
                citations: result.citations,
                warnings: result.warnings,
                created_at: new Date().toISOString(),
            };
            setMessages((current) => [...current, assistantMessage]);
            setDraft("");
            if (!sessionId) {
                setSearchParams({ session_id: String(result.session_id) });
            }
        } catch (sendError: unknown) {
            setError(apiError(sendError));
            setMessages((current) =>
                current.filter((item) => item.id !== temporaryMessage.id)
            );
        } finally {
            setSending(false);
        }
    }

    return (
        <section className="mx-auto flex min-h-[calc(100vh-5rem)] max-w-4xl flex-col p-6">
            <h1 className="mb-2 text-2xl font-bold">Dietary Assistant</h1>
            <p className="mb-5 text-gray-600">
                Answers use retrieved evidence and your saved dietary constraints.
                This assistant does not replace professional medical advice.
            </p>

            {error && (
                <p role="alert" className="mb-4 rounded bg-red-100 p-3 text-red-700">
                    {error}
                </p>
            )}

            <div
                aria-live="polite"
                className="mb-4 flex-1 space-y-4 overflow-y-auto rounded-lg bg-gray-100 p-4"
            >
                {loadingHistory && <p>Loading conversation...</p>}
                {!messages.length && !loadingHistory && (
                    <p className="text-gray-600">
                        Ask about food nutrition, dietary guidance, or meal ideas.
                    </p>
                )}
                {messages.map((message) => (
                    <article
                        key={message.id}
                        className={`max-w-[90%] rounded-lg p-4 ${
                            message.role === "user"
                                ? "ml-auto bg-blue-600 text-white"
                                : "bg-white text-gray-900 shadow-sm"
                        }`}
                    >
                        <p className="mb-1 text-xs font-semibold uppercase opacity-70">
                            {message.role === "user" ? "You" : "Assistant"}
                        </p>
                        <p className="whitespace-pre-wrap">{message.content}</p>
                        {message.role === "assistant" && (
                            <>
                                {!!message.warnings?.length && (
                                    <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-amber-800">
                                        {message.warnings.map((warning) => (
                                            <li key={warning}>{warning}</li>
                                        ))}
                                    </ul>
                                )}
                                <Citations items={message.citations} />
                            </>
                        )}
                    </article>
                ))}
            </div>

            <form onSubmit={handleSubmit} className="flex gap-3">
                <label htmlFor="chat-message" className="sr-only">
                    Your dietary question
                </label>
                <textarea
                    id="chat-message"
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    maxLength={4000}
                    rows={2}
                    placeholder="Ask a dietary question..."
                    className="min-w-0 flex-1 resize-y rounded border border-gray-300 bg-white px-3 py-2"
                    required
                />
                <button
                    type="submit"
                    disabled={sending || !draft.trim()}
                    className="self-end rounded bg-blue-600 px-5 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                >
                    {sending ? "Thinking..." : "Send"}
                </button>
            </form>
        </section>
    );
}

export default Chat;
