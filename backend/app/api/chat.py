from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, noload

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.knowledge import ChatMessage, ChatSession
from app.models.user import User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
)
from app.services.chat_service import (
    create_chat_response,
    get_user_session,
    save_chat_turn,
)


router = APIRouter(prefix="/api/chat", tags=["Dietary assistant"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    if request.session_id is None:
        session = ChatSession(
            user_id=current_user.id,
            title=request.message.strip()[:180] or "Dietary question",
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    else:
        session = get_user_session(
            db,
            user_id=current_user.id,
            session_id=request.session_id,
        )

    db.add(
        ChatMessage(
            session_id=session.id,
            role="user",
            content=request.message.strip(),
        )
    )
    db.commit()

    response = create_chat_response(
        db,
        user=current_user,
        session=session,
        message=request.message.strip(),
    )
    save_chat_turn(db, session=session, response=response)
    return response


@router.get("/history", response_model=list[ChatSessionResponse])
def list_chat_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatSession]:
    return db.scalars(
        select(ChatSession)
        .options(noload(ChatSession.messages))
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(50)
    ).all()


@router.get(
    "/history/{session_id}",
    response_model=ChatSessionResponse,
)
def get_chat_history(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatSession:
    session = get_user_session(
        db,
        user_id=current_user.id,
        session_id=session_id,
    )
    if not session.messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return session
