import api from "./api";

export interface ChatCitation {
    evidence_id: string;
    source_title: string;
    source_url: string | null;
    source_type: string;
    authority_score: number;
    excerpt: string;
}

export interface ChatMessage {
    id: number;
    role: "user" | "assistant";
    content: string;
    citations: ChatCitation[];
    warnings: string[];
    confidence: number | null;
    created_at: string;
}

export interface ChatSession {
    id: number;
    title: string;
    summary: string;
    created_at: string;
    updated_at: string;
    messages: ChatMessage[];
}

export interface ChatResponse {
    session_id: number;
    answer: string;
    citations: ChatCitation[];
    confidence: number;
    warnings: string[];
    intent: string;
}

export async function sendMessage(
    message: string,
    sessionId?: number
): Promise<ChatResponse> {
    const response = await api.post<ChatResponse>("/api/chat", {
        message,
        ...(sessionId ? { session_id: sessionId } : {}),
    });
    return response.data;
}

export async function getChatHistory(): Promise<ChatSession[]> {
    const response = await api.get<ChatSession[]>("/api/chat/history");
    return response.data;
}

export async function getChatSession(id: number): Promise<ChatSession> {
    const response = await api.get<ChatSession>(
        `/api/chat/history/${id}`
    );
    return response.data;
}
