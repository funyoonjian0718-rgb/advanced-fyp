from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.user_profile import UserProfile
from app.schemas.profile import UserProfileResponse, UserProfileUpdate


router = APIRouter(
    prefix="/api/profile",
    tags=["Profile"],
)


@router.get("", response_model=UserProfileResponse)
def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfile:
    profile = db.scalar(
        select(UserProfile).where(UserProfile.user_id == current_user.id)
    )

    if profile is None:
        # Backward compatibility for accounts created before automatic profile
        # creation was added. Do not fabricate any personal/dietary values.
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
        db.commit()
        db.refresh(profile)

    return profile


@router.put("", response_model=UserProfileResponse)
def update_profile(
    request: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfile:
    profile = db.scalar(
        select(UserProfile).where(UserProfile.user_id == current_user.id)
    )

    if profile is None:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)

    for field, value in request.model_dump(exclude_unset=True).items():
        setattr(profile, field, value)

    profile.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(profile)

    return profile
