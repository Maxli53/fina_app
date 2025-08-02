"""
Authentication models
"""
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class User(BaseModel):
    """User model for authentication"""
    id: int
    email: EmailStr
    username: str
    is_active: bool = True
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """User creation schema"""
    email: EmailStr
    username: str
    password: str


class UserLogin(BaseModel):
    """User login schema"""
    username: str
    password: str


class Token(BaseModel):
    """JWT token schema"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    user_id: Optional[int] = None