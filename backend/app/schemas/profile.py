from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserProfileUpdate(BaseModel):
    full_name: str | None = Field(default=None, max_length=150)
    age: int | None = Field(default=None, ge=1, le=120)
    gender: str | None = Field(default=None, max_length=50)
    height_cm: float | None = Field(default=None, gt=0, le=300)
    weight_kg: float | None = Field(default=None, gt=0, le=700)
    activity_level: str | None = Field(default=None, max_length=50)
    dietary_preference: str | None = Field(default=None, max_length=100)
    allergies: str | None = Field(default=None, max_length=2000)

    model_config = ConfigDict(extra="forbid")


class UserProfileResponse(BaseModel):
    id: int
    user_id: int
    full_name: str | None
    age: int | None
    gender: str | None
    height_cm: float | None
    weight_kg: float | None
    activity_level: str | None
    dietary_preference: str | None
    allergies: str | None
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
