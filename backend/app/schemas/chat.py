from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: int | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must contain non-whitespace characters.")
        return value


class ChatCitation(BaseModel):
    evidence_id: str
    source_title: str
    source_url: str | None
    source_type: str
    authority_score: float
    excerpt: str


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    citations: list[ChatCitation]
    confidence: float
    warnings: list[str]
    intent: str


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    citations: list[ChatCitation]
    warnings: list[str] = Field(default_factory=list)
    confidence: float | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatSessionResponse(BaseModel):
    id: int
    title: str
    summary: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
