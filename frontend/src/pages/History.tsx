import { useEffect, useState } from "react";
import { isAxiosError } from "axios";
import { Link } from "react-router-dom";

import { getChatHistory, type ChatSession } from "../services/chat";

function History() {
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let active = true;

        async function loadHistory() {
            try {
                const history = await getChatHistory();
                if (active) {
                    setSessions(history);
                }
            } catch (loadError: unknown) {
                if (active) {
                    const detail = isAxiosError<{ detail?: unknown }>(loadError)
                        ? loadError.response?.data?.detail
                        : undefined;
                    setError(
                        typeof detail === "string"
                            ? detail
                            : "Failed to load your conversation history."
                    );
                }
            } finally {
                if (active) {
                    setLoading(false);
                }
            }
        }

        void loadHistory();
        return () => {
            active = false;
        };
    }, []);

    return (
        <section className="mx-auto max-w-4xl p-6">
            <h1 className="mb-5 text-2xl font-bold">Query History</h1>

            {error && (
                <p role="alert" className="mb-4 rounded bg-red-100 p-3 text-red-700">
                    {error}
                </p>
            )}
            {loading && <p>Loading conversation history...</p>}
            {!loading && !sessions.length && !error && (
                <p className="text-gray-600">
                    Your previous dietary questions will appear here.
                </p>
            )}

            <ul className="space-y-3">
                {sessions.map((session) => (
                    <li key={session.id}>
                        <Link
                            to={`/chat?session_id=${session.id}`}
                            className="block rounded-lg bg-white p-4 shadow hover:bg-gray-50"
                        >
                            <span className="font-medium">{session.title}</span>
                            <span className="mt-1 block text-sm text-gray-500">
                                Updated{" "}
                                {new Date(session.updated_at).toLocaleString()}
                            </span>
                        </Link>
                    </li>
                ))}
            </ul>
        </section>
    );
}

export default History;
