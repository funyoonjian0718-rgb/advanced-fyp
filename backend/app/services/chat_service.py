from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge import ChatMessage, ChatSession
from app.models.user import User
from app.models.user_profile import UserProfile
from app.schemas.chat import ChatResponse
from app.services.constraints import ConstraintResult, enforce_constraints
from app.services.generation import (
    EvidenceVerificationError,
    ModelOutputError,
    VerifiedAnswer,
    generate_meal_plan_answer,
    generate_verified_answer,
    is_meal_plan_request,
)
from app.services.query_processor import process_query
from app.services.retrieval import (
    compress_evidence,
    rerank_evidence,
    retrieve_evidence,
    retrieve_meal_plan_foods,
)

'''
process the user's question through the complete RAG pipeline

Pipeline:
1. Load the user's dietary profile
2. process the question and identify the constrains
3. retrieve the relevant knowledge
4. filter the non-compatible food recommendation
5.reeank and compress the evidence
6. generate the answer using the local LLM
7. verify the generated claims againts the evidence
'''
def _safe_refusal(constraints: ConstraintResult) -> VerifiedAnswer:
    warnings = list(constraints.warnings)
    if constraints.has_known_conflict and (
        constraints.allergen_status_unknown or constraints.dietary_status_unknown
    ):
        answer = (
            "I can't recommend the retrieved foods: some conflict with your "
            "dietary or allergen restrictions, and the compatibility of others "
            "could not be verified from the available records."
        )
    elif constraints.has_known_conflict:
        answer = (
            "I can't recommend the retrieved foods because they conflict with "
            "your dietary or allergen restrictions."
        )
    elif constraints.allergen_status_unknown:
        answer = (
            "I can't verify that the retrieved foods are safe for your allergy "
            "because their allergen information is missing or unclear. Please "
            "check the food label or consult a qualified dietitian."
        )
    elif constraints.dietary_status_unknown:
        answer = (
            "I can't verify that the retrieved foods meet your dietary "
            "requirements from the available records."
        )
    else:
        answer = (
            "I do not have enough verified evidence to answer that safely. "
            "Please consult a qualified health professional for personal medical "
            "advice."
        )
        warnings.append(
            "The available knowledge base did not contain enough compatible "
            "evidence to support a recommendation."
        )
    return VerifiedAnswer(
        answer=answer,
        citations=[],
        warnings=warnings,
        supported_claims=0,
        rejected_claims=0,
    )


def create_chat_response(
    db: Session,
    *,
    user: User,
    session: ChatSession,
    message: str,
) -> ChatResponse:
    #load the dietary profile so that the preferences and allergies
    #can be considered when processign the question
    profile = db.scalar(
        select(UserProfile).where(UserProfile.user_id == user.id)
    )
    # Analyse the question to determine:
    # - the user's intent
    # - dietary restrictions
    # - allergies
    # - whether structured nutrition data is needed
    # - whether guideline/document evidence is needed
    # - whether the user is requesting a recommendation
    plan = process_query(
        message,
        dietary_preference=profile.dietary_preference if profile else None,
        allergies=profile.allergies if profile else None,
        activity_level=profile.activity_level if profile else None,
    )
    if plan.intent == "out_of_scope":
        return ChatResponse(
            session_id=session.id,
            answer=(
                "That looks like a terminal command, not a nutrition question. "
                "Run it in the backend PowerShell terminal; this chat is for "
                "dietary and nutrition questions."
            ),
            citations=[],
            warnings=[],
            intent=plan.intent,
        )
    try:
        # Retrieve candidate evidence using the query plan.
        # The retrieval stage combines lexical BM25 search with
        # dense vector search from Qdrant.
        if is_meal_plan_request(message) and not plan.constraints.numeric_constraints:
            retrieved = retrieve_meal_plan_foods(db, plan)
        else:
            retrieved = retrieve_evidence(db, plan)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except (OSError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Local retrieval models or the vector index are unavailable. "
                "Verify ingestion, model cache files, and the embedding settings."
            ),
        ) from error

    # Apply deterministic dietary and allergy constraints before
    # sending evidence to the LLM.
    #
    # The LLM should not be responsible for deciding whether a food
    # violates an allergy or dietary restriction.
    constrained = enforce_constraints(retrieved, plan)

    has_numeric_food_matches = any(item.kind == "food" for item in retrieved)
    if (
        plan.constraints.numeric_constraints
        and not has_numeric_food_matches
        and not plan.document_lookup
    ):
        verified = VerifiedAnswer(
            answer=(
                "No foods in the available dataset match the requested "
                "nutrition conditions."
            ),
            citations=[],
            warnings=[],
            supported_claims=0,
            rejected_claims=0,
        )
    elif (
        plan.recommendation_requested
        and (
            plan.constraints.diets
            or plan.constraints.allergens
            or plan.constraints.unknown_dietary_constraint
            or plan.constraints.unknown_allergen_constraint
        )
        and not any(item.kind == "food" for item in constrained.allowed)
    ):
        verified = _safe_refusal(constrained)
    elif not constrained.allowed:
        verified = _safe_refusal(constrained)
    elif (
    is_meal_plan_request(message)
    and not plan.constraints.numeric_constraints
    ):
        # Meal plans are composed from already-filtered structured food records.
        # This avoids asking the small local LLM to invent a multi-item answer
        # that the strict single-claim verifier cannot validate.
        verified = generate_meal_plan_answer(
            message,
            constrained.allowed,
            constrained,
        )
    elif plan.structured_lookup and not plan.constraints.numeric_constraints:
        verified = generate_verified_answer(
            message,
            plan,
            constrained.allowed,
            constrained,
        )
    else:
        rerank_query = message
        if plan.constraints.diets or plan.constraints.allergens:
            rerank_query += (
                " Requirements: "
                + ", ".join((*plan.constraints.diets, *plan.constraints.allergens))
            )
        try:
            # Rerank the retrieved candidates using a CrossEncoder model.
            # This gives the system a second-stage relevance score after
            # the initial BM25 and dense retrieval.
            reranked = rerank_evidence(rerank_query, constrained.allowed)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The local reranker is unavailable. Verify the model cache "
                    "and reranker settings."
                ),
            ) from error
        compressed = compress_evidence(rerank_query, reranked)
        try:
            # Generate an answer using only the retrieved evidence.
            # The generation service also validates the model's claims
            # against the evidence before returning the final response.
            verified = generate_verified_answer(
                message,
                plan,
                compressed,
                constrained,
            )
        except ModelOutputError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(error),
            ) from error
        except EvidenceVerificationError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The local language model is unavailable. Verify the model "
                    "files, GPU memory, and backend virtual-environment packages."
                ),
            ) from error

    return ChatResponse(
        session_id=session.id,
        answer=(
            "No foods in the available dataset match the requested nutrition "
            "conditions. "
            + verified.answer
            if plan.constraints.numeric_constraints
            and not has_numeric_food_matches
            and plan.document_lookup
            else verified.answer
        ),
        citations=verified.citations,
        warnings=verified.warnings,
        intent=plan.intent,
    )


def save_chat_turn(
    db: Session,
    *,
    session: ChatSession,
    response: ChatResponse,
) -> None:
    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=response.answer,
            citations=[citation.model_dump() for citation in response.citations],
            warnings=response.warnings,
        )
    )
    db.flush()
    recent = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(6)
    ).all()
    session.summary = "\n".join(
        f"{item.role}: {item.content[:280]}"
        for item in reversed(recent)
    )[-1400:]
    session.updated_at = datetime.now(timezone.utc)
    db.commit()


def get_user_session(
    db: Session,
    *,
    user_id: int,
    session_id: int,
) -> ChatSession:
    session = db.scalar(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return session
